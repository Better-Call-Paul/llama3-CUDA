/*
 * llama3.cpp is a C++/CUDA implementation of the Llama 3 model.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cctype>
#include <ctime>
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <mma.h>

// Error checking macro for CUDA calls
#define CUDA_CHECK(val) { \
    if (val != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", val, cudaGetErrorString(val), __FILE__, __LINE__); \
        fflush(stderr); \
        exit(val); \
    } \
}

using namespace nvcuda; // just for tensor core usage 

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define PADDING_SIZE 1

struct __align__(16) Align16 {
    float x, y, z, w;
};

struct __align__(8) Align8 {
    float a, b;
};

// ----------------------------------------------------------------------------
// Transformer model
// ----------------------------------------------------------------------------
struct Config {
    int dim;            // D
    int hidden_dim;     // DD
    int n_layers;       // NL
    int n_heads;        // QHN, HN, HD = 48
    int n_kv_heads;     // KVHN = 6
    int vocab_size;     // VS
    int max_seq_len;    // M
};

// The TransformerWeights structure will hold pointers to GPU memory
struct TransformerWeights {
    float *token_embedding;     // (VS, D)
    float *rms_att_weight;      // (NL, D)
    float *wq;                  // (NL, D, D)
    float *wk;                  // (NL, D, D)
    float *wv;                  // (NL, D, D)
    float *wo;                  // (NL, D, D)
    float *rms_ffn_weight;      // (NL, D)
    float *w1;                  // (NL, DD, D)
    float *w2;                  // (NL, D, DD)
    float *w3;                  // (NL, DD, D)
    float *rms_final_weight;    // (D,)
    // (optional) classifier weights for the logits, on the last layer
    float *wcls;
};

// The RunState structure will hold pointers to GPU memory
struct RunState {
    // Current wave of activations
    float *x;           // (D,) activation at current time stamp
    float *xb;          // (D,) same, but inside a residual branch
    float *xb2;         // (D,) an additional buffer just for convenience
    float *hb;          // (DD,) buffer for hidden dimension in the ffn
    float *hb2;         // (DD,) buffer for hidden dimension in the ffn
    float *q;           // (D,) query
    float *k;           // (D,) key
    float *v;           // (D,) value
    float *att;         // (HN, M) buffer for scores/attention values
    float *logits_gpu;  // Output logits in GPU
    float *logits;      // Output logits in CPU
    // KV cache
    float *key_cache;   // (NL, M, D)
    float *value_cache; // (NL, M, D)
};

// Transformer structure
struct Transformer {
    Config config;              // Hyperparameters of the architecture
    TransformerWeights weights; // Weights of the model
    RunState state;             // Buffers for the "wave" of activations
    // Memory mapping details
    int fd;                     // File descriptor for memory mapping
    float *data;                // Memory mapped data pointer
    ssize_t file_size;          // Size of the checkpoint file in bytes
};

void malloc_run_state(RunState &s, const Config &p) {
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    CUDA_CHECK(cudaMalloc(&s.x, p.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.xb, p.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.xb2, p.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.hb, p.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.hb2, p.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.q, p.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.key_cache, p.n_layers * p.max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.value_cache, p.n_layers * p.max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.att, p.n_heads * p.max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s.logits_gpu, p.vocab_size * sizeof(float)));
    // Allocate CPU memory for logits
    s.logits = new float[p.vocab_size]();

    // Ensure all cudaMallocs went fine
    if (!s.x || !s.xb || !s.xb2 || !s.hb || !s.hb2 || !s.q
        || !s.key_cache || !s.value_cache || !s.att || !s.logits_gpu || !s.logits) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState &s) {
    CUDA_CHECK(cudaFree(s.x));
    CUDA_CHECK(cudaFree(s.xb));
    CUDA_CHECK(cudaFree(s.xb2));
    CUDA_CHECK(cudaFree(s.hb));
    CUDA_CHECK(cudaFree(s.hb2));
    CUDA_CHECK(cudaFree(s.q));
    CUDA_CHECK(cudaFree(s.att));
    CUDA_CHECK(cudaFree(s.logits_gpu));
    delete[] s.logits;
    CUDA_CHECK(cudaFree(s.key_cache));
    CUDA_CHECK(cudaFree(s.value_cache));
}

void memory_map_weights(TransformerWeights &w, const Config &p, float *ptr, int shared_weights) {
    int head_size = p.dim / p.n_heads;
    // Make sure the multiplications below are done in 64-bit to fit the parameter counts of large models
    unsigned long long n_layers = p.n_layers;
    w.token_embedding = ptr;
    ptr += p.vocab_size * p.dim;
    w.rms_att_weight = ptr;
    ptr += n_layers * p.dim;
    w.wq = ptr;
    ptr += n_layers * p.dim * p.dim;
    w.wk = ptr;
    ptr += n_layers * p.dim * p.dim;
    w.wv = ptr;
    ptr += n_layers * p.dim * p.dim;
    w.wo = ptr;
    ptr += n_layers * p.dim * p.dim;
    w.rms_ffn_weight = ptr;
    ptr += n_layers * p.dim;
    w.w1 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.w2 = ptr;
    ptr += n_layers * p.hidden_dim * p.dim;
    w.w3 = ptr;
    ptr += n_layers * p.dim * p.hidden_dim;
    w.rms_final_weight = ptr;
    ptr += p.dim;
    ptr += p.max_seq_len * head_size / 2; // Skip freq_cis_real
    ptr += p.max_seq_len * head_size / 2; // Skip freq_cis_imag
    w.wcls = shared_weights ? w.token_embedding : ptr;
}

void read_checkpoint(const char *checkpoint, Config &config, TransformerWeights &weights,
                     int &fd, float *&data, ssize_t &file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        printf("Error opening file: %s\n", strerror(errno));
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // Read in the config header
    if (fread(&config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // Negative vocab size indicates unshared weights
    int shared_weights = config.vocab_size > 0 ? 1 : 0;
    config.vocab_size = abs(config.vocab_size);
    // Figure out the file size
    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fclose(file);
    // Memory map the Transformer weights
    fd = open(checkpoint, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    data = (float *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    // Allocate & copy mmap data to the GPU
    float *weights_ptr;
    size_t weights_size = file_size - sizeof(Config);
    CUDA_CHECK(cudaMalloc(&weights_ptr, weights_size));
    CUDA_CHECK(cudaMemcpy(weights_ptr, data + sizeof(Config) / sizeof(float), weights_size, cudaMemcpyHostToDevice));
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer &t, const char *checkpoint_path) {
    // Read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, t.config, t.weights, t.fd, t.data, t.file_size);
    // Allocate the RunState buffers
    malloc_run_state(t.state, t.config);
}

void free_transformer(Transformer &t) {
    // Close the memory mapping
    if (t.data != MAP_FAILED) { munmap(t.data, t.file_size); }
    if (t.fd != -1) { close(t.fd); }
    // Free the weights allocated on the GPU
    CUDA_CHECK(cudaFree(t.weights.token_embedding));
    // Free the RunState buffers
    free_run_state(t.state);
}

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

#define CEIL_DIV(M, N) ((M + N - 1) / N)

const int num_threads_large = 1024;
const int num_threads_small = 64;

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size, int elementsPerThread) {
    // Parallel reduction of sum of squares via CUB
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size)
            ss += x[j] * x[j];
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    // Calculate normalization factor
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // Normalize and scale
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}

void rmsnorm(float *o, float *x, float *weight, int size) {
    int elementsPerThread = CEIL_DIV(size, num_threads_large);
    rmsnorm_kernel<<<1, num_threads_large>>>(o, x, weight, size, elementsPerThread);
}

template<typename T>
__inline__ __device__ void warp_reduce_sum(T& curr_sum, const uint thread_group_width = WARP_SIZE) {
    
    #pragma unroll
    for (uint i = thread_group_width / 2; i > 0; i >>= 1) {
        curr_sum += __shfl_xor_sync(FULL_MASK, curr_sum, i, WARP_SIZE);
    }
}

// block reduction for threads that have the same threadIdx.y reduce values
template<typename T, size_t ROWS = 1>
__inline__ __device__ void block_reduce_sum(T& curr_sum) {

    __shared__ T shmem[ROWS][WARP_SIZE + PADDING_SIZE];
    const uint lane = threadIdx.x % (WARP_SIZE + PADDING_SIZE);
    const uint warp_id = threadIdx.x / (WARP_SIZE + PADDING_SIZE);

    warp_reduce_sum<T>(curr_sum);

    // first of every warp loads its partial value
    if (lane == 0) {
        shmem[threadIdx.y][warp_id] = curr_sum;
    }

    __syncthreads();

    bool is_valid = threadIdx.x < (blockDim.x / 32.0f);
    curr_sum = is_valid ? shmem[threadIdx.y][lane] : T(0.0f); 

    // first warp then reduces all partial sums in other warps
    if (warp_id == 0) {
        warp_reduce_sum<T>(curr_sum);
    }
}

template<typename T>
__inline__ __device__ void warp_reduce_max(T& curr_max, const uint thread_group_width = WARP_SIZE) {

    #pragma unroll
    for (uint i = thread_group_width / 2; i > 0; i >>= 1) {
        curr_max = max(curr_max, __shfl_xor_sync(FULL_MASK, curr_max, i, WARP_SIZE));
    }
}

template<typename T, size_t ROWS = 1>
__inline__ __device__ void block_reduce_max(T& curr_max) {

    __shared__ T shmem[ROWS][WARP_SIZE + PADDING_SIZE];
    const uint warp_id = threadIdx.x / (WARP_SIZE + PADDING_SIZE);
    const uint lane = threadIdx.x % (WARP_SIZE + PADDING_SIZE);

    warp_reduce_max<T>(curr_max);

    if (lane == 0) {
        shmem[threadIdx.y][warp_id] = curr_max;
    }

    __syncthreads();

    bool is_valid = threadIdx.x < (blockDim.x / 32.0f);
    curr_max = is_valid ? shmem[threadIdx.y][lane] : (T)(0.0f);

    if (warp_id == 0) {
        warp_reduce_max<T>(curr_max);
    }
}

// small softmax
template<typename T, typename activation_T, int cols_per_thread>
__global__ void softmax_local(int M, int N, const T *input, T *output) {
    
    // Number of activation_T elements that fit into T
    const uint pack_size = sizeof(T) / sizeof(activation_T);
    const uint num_packs = CEIL_DIV(cols_per_thread, pack_size);
    
    // Starting row index for this thread
    const uint m_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const uint thread_id = threadIdx.x;

    // Local buffer to store intermediate activation values
    float buf[cols_per_thread];
    
    // Iterate over rows assigned to this thread
    #pragma unroll
    for (uint row = m_idx; row < M; row += gridDim.x * blockDim.y) { 
        // Calculate the offset for the current row in the flattened input/output arrays
        const uint row_offset = row * ((N + pack_size - 1) / pack_size);
        const T *row_input = input + row_offset;
        T *row_output = output + row_offset;
        
        // Initialize local maximum for numerical stability
        float local_max = -INFINITY;
        
        // Load and process each pack of activation_T elements
        #pragma unroll
        for (uint i = 0; i < num_packs; ++i) {
            // Calculate the column index this thread is responsible for
            const uint col = i * blockDim.x + thread_id;
            T tmp_in = row_input[col];
            const activation_T *pack_input = reinterpret_cast<const activation_T*>(&tmp_in);
            
            if (col < N / pack_size) {
                // Load each activation_T element, convert to float, and update local_max
                #pragma unroll
                for (int j = 0; j < pack_size; j++) {
                    buf[i * pack_size + j] = static_cast<float>(pack_input[j]);
                    local_max = max(local_max, buf[i * pack_size + j]);
                }
            } else {
                // Handle out-of-bounds columns by setting to -INFINITY
                #pragma unroll
                for (int j = 0; j < pack_size; j++) {
                    buf[i * pack_size + j] = -INFINITY;
                }
            }
        }
        
        // Perform warp-level reduction to find the global maximum in the row
        warp_reduce_max<float>(local_max, blockDim.x);
        
        // Initialize local sum for normalization
        float local_sum = 0.0f;
        
        // Compute exponentials and accumulate the sum
        #pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            buf[i] = expf(buf[i] - local_max);
            local_sum += buf[i];
        }
        
        // Perform warp-level reduction to find the total sum of exponentials
        warp_reduce_sum<float>(local_sum, blockDim.x);
        
        // Normalize the exponentials to obtain softmax probabilities
        T tmp_out;
        activation_T *pack_output = reinterpret_cast<activation_T*>(&tmp_out);
        
        #pragma unroll
        for (uint i = 0; i < num_packs; ++i) {
            const uint col = i * blockDim.x + thread_id;
            if (col < N / pack_size) {
                // Normalize each activation_T element and store in the output buffer
                #pragma unroll
                for (int j = 0; j < pack_size; j++) {
                    pack_output[j] = static_cast<activation_T>(buf[i * pack_size + j] / local_sum);
                }
                row_output[col] = tmp_out;
            }
        }
    }
}



__device__ void softmax_gpu(float *__restrict__ x, int size) {
    int tid = threadIdx.x;
    int step = blockDim.x;

    // Find max value (for numerical stability)
    float max_val = tid < size ? x[tid] : -INFINITY;
    for (int i = tid + step; i < size; i += step) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    // Exp and sum
    float sum = 0.0f;
    for (int i = tid; i < size; i += step) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    // Normalize
    for (int i = tid; i < size; i += step) {
        x[i] /= sum;
    }
}

template<const uint BM, const uint BN, const uint BK>
__global__ void wmma_gemm(int M, int N, int K, float alpha, const float *x, const float *w, float beta, float *xout) {
    // Compute block row and column
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // Shared memory for A and B tiles
    extern __shared__ half shared_mem[];
    half* A_Shmem = shared_mem;
    half* B_Shmem = A_Shmem + BM * BK;

    // Thread ID within the block
    int threadId = threadIdx.x + threadIdx.y * blockDim.x;

    // Number of threads per block
    int numThreads = blockDim.x * blockDim.y;

    // Initialize accumulator fragment
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over BK tiles
    for (int bk = 0; bk < K; bk += BK) {
        // **Vectorized Load of A with float4**
        int numAElements = BM * BK;
        int numAFloat4 = (numAElements + 3) / 4; // Ceiling division

        for (int idx = threadId; idx < numAFloat4; idx += numThreads) {
            int idx4 = idx * 4; // Starting index for float4 load
            int tile_row = idx4 / BK;
            int tile_col = idx4 % BK;

            int global_row = blockRow + tile_row;
            int global_col = bk + tile_col;

            float4 data;

            // Boundary check for global memory access
            if (global_row < M && (global_col + 3) < K) {
                const float4* src_ptr = reinterpret_cast<const float4*>(&x[global_row * K + global_col]);
                data = *src_ptr;
            } else {
                // Handle boundary conditions
                float tmp[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4; ++i) {
                    int col = global_col + i;
                    if (global_row < M && col < K) {
                        tmp[i] = x[global_row * K + col];
                    }
                }
                data = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
            }

            // Convert to half and store in shared memory
            int shmem_idx = tile_row * BK + tile_col;
            half* shmem_ptr = &A_Shmem[shmem_idx];
            shmem_ptr[0] = __float2half(data.x);
            if (tile_col + 1 < BK) shmem_ptr[1] = __float2half(data.y);
            if (tile_col + 2 < BK) shmem_ptr[2] = __float2half(data.z);
            if (tile_col + 3 < BK) shmem_ptr[3] = __float2half(data.w);
        }

        // **Vectorized Load of B with float4**
        int numBElements = BK * BN;
        int numBFloat4 = (numBElements + 3) / 4;

        for (int idx = threadId; idx < numBFloat4; idx += numThreads) {
            int idx4 = idx * 4;
            int tile_row = idx4 % BK;
            int tile_col = idx4 / BK;

            int global_row = bk + tile_row;
            int global_col = blockCol + tile_col * 4;

            float4 data;

            if (global_row < K && (global_col + 3) < N) {
                const float4* src_ptr = reinterpret_cast<const float4*>(&w[global_row * N + global_col]);
                data = *src_ptr;
            } else {
                float tmp[4] = {0, 0, 0, 0};
                for (int i = 0; i < 4; ++i) {
                    int col = global_col + i;
                    if (global_row < K && col < N) {
                        tmp[i] = w[global_row * N + col];
                    }
                }
                data = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
            }

            // Convert to half and store in shared memory (column-major order)
            int shmem_idx = tile_col * BK + tile_row;
            half* shmem_ptr = &B_Shmem[shmem_idx];
            shmem_ptr[0] = __float2half(data.x);
            if (tile_row + 1 < BK) shmem_ptr[1] = __float2half(data.y);
            if (tile_row + 2 < BK) shmem_ptr[2] = __float2half(data.z);
            if (tile_row + 3 < BK) shmem_ptr[3] = __float2half(data.w);
        }


        __syncthreads();

        // Compute matrix multiplication using WMMA
        int warpId = threadId / warpSize;
        int laneId = threadId % warpSize;

        // Compute warp's position within the block
        int warpsPerBlockM = BM / 16;
        int warpsPerBlockN = BN / 16;
        int warpM = warpId / warpsPerBlockN;
        int warpN = warpId % warpsPerBlockN;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;

        const half* a_shmem_ptr = &A_Shmem[warpM * 16 * BK];
        const half* b_shmem_ptr = &B_Shmem[warpN * 16 * BK];

        wmma::load_matrix_sync(a_frag, a_shmem_ptr, BK);
        wmma::load_matrix_sync(b_frag, b_shmem_ptr, BK);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    // Store the result back to global memory with alpha and beta scaling
    int warpId = threadId / warpSize;
    int laneId = threadId % warpSize;

    int warpsPerBlockM = BM / 16;
    int warpsPerBlockN = BN / 16;
    int warpM = warpId / warpsPerBlockN;
    int warpN = warpId % warpsPerBlockN;

    int cRow = blockRow + warpM * 16;
    int cCol = blockCol + warpN * 16;

    if (cRow < M && cCol < N) {
        float* c_ptr = &xout[cRow * N + cCol];

        // Load original C matrix
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_orig_frag;
        wmma::load_matrix_sync(c_orig_frag, c_ptr, N, wmma::mem_row_major);

        // Apply alpha and beta scaling
        for (int i = 0; i < c_frag.num_elements; ++i) {
            c_frag.x[i] = alpha * c_frag.x[i] + beta * c_orig_frag.x[i];
        }

        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    }
}

__global__ void matmul_kernel(float *xout, float *x, float *w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d)
        return;

    float sum = 0.0f;
    for (int j = 0; j < n; j++) {
        sum += w[i * n + j] * x[j];
    }
    xout[i] = sum;
}

void matmul(float *xout, float *x, float *w, int N, int M) {
    // let x be KxN
    // let w be MxK
    // n = N
    // d = M
    // k = N
    const uint BK = 16;
    
    if (N >= 128 && M >= 128) {
        const uint BN = 128;
        const uint BM = 128;

        const uint warpsPerBlockM = BM / 16;
        const uint warpsPerBlockN = BN / 16;
        const uint totalWarpsPerBlock = warpsPerBlockM * warpsPerBlockN;

        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim(32, totalWarpsPerBlock);

        size_t sharedMemSize = (BM * BK + BK * BN) * sizeof(half);

        wmma_gemm<BN, BM, BK><<<gridDim, blockDim, sharedMemSize>>>(M, N, N, 1.0f, w, x, 1.0f, xout);
    }
    else {
        const uint BN = 64;
        const uint BM = 64;

        const uint warpsPerBlockM = BM / 16;
        const uint warpsPerBlockN = BN / 16;
        const uint totalWarpsPerBlock = warpsPerBlockM * warpsPerBlockN;

        dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
        dim3 blockDim(32, totalWarpsPerBlock);

        size_t sharedMemSize = (BM * BK + BK * BN) * sizeof(half);

        wmma_gemm<BN, BM, BK><<<gridDim, blockDim, sharedMemSize>>>(M, N, N, 1.0f, w, x, 1.0f, xout);
    }

    //wmma_gemm<BM, BN, BK>
    //<<<CEIL_DIV(d, num_threads_small), num_threads_small>>>(d, n, n, 1.0f, w, x, 1.0f, xout);
    //matmul_kernel<<<CEIL_DIV(d, num_threads_small), num_threads_small>>>(xout, x, w, n, d);
}

// Additional neural net blocks
__global__ void RoPe_rotation_kernel(int pos, float *sq, float *sk, int kv_dim, int head_size) {
    int i = threadIdx.x * 2;
    if (i >= kv_dim)
        return;
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / static_cast<float>(head_size));
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        float *vec = v == 0 ? sq : sk; // The vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

void RoPe_rotation(int pos, RunState &s, int dim, int kv_dim, int head_size) {
    RoPe_rotation_kernel<<<1, dim / 2>>>(pos, s.q, s.k, kv_dim, head_size);
}

__global__ void multi_head_attention_kernel(int pos, int seq_len, float *sq, float *satt, float *sxb, float *key_cache,
                                            float *value_cache, int kv_dim, int kv_mul, int head_size, int loff) {
    int h = blockIdx.x;
    // Get the query vector for this head
    float *q = sq + h * head_size;
    // Attention scores for this head
    float *att = satt + h * seq_len;
    // Iterate over all timesteps, including the current one
    for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
        // Get the key vector for this head and at this timestep
        float *k = key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        // Calculate the attention score as the dot product of q and k
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) {
            score += q[i] * k[i];
        }
        score /= sqrtf(static_cast<float>(head_size));
        // Save the score to the attention buffer
        att[t] = score;
    }
    __syncthreads();

    // Softmax the scores to get attention weights
    softmax_gpu(att, pos + 1);
    __syncthreads();

    // Weighted sum of the values, store back into xb
    float *xb = sxb + h * head_size;
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int t = 0; t <= pos; t++) {
            // Get the value vector for this head and at this timestep
            float *v = value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
            // Get the attention weight for this timestep
            float a = att[t];
            val += a * v[i];
        }
        xb[i] = val;
    }
}

void multi_head_attention(int pos, const Config &p, RunState &s, int kv_dim, int kv_mul, int head_size, int loff) {
    multi_head_attention_kernel<<<p.n_heads, num_threads_large>>>(pos, p.max_seq_len, s.q, s.att, s.xb,
                                                                   s.key_cache, s.value_cache, kv_dim, kv_mul,
                                                                   head_size, loff);
}

__global__ void f_silu_elementwise_mul_w3_kernel(float *shb, float *shb2, int hidden_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_dim) {
        float val = shb[i];
        // SiLU activation function
        val *= (1.0f / (1.0f + expf(-val)));
        // Elementwise multiply with w3(x)
        val *= shb2[i];
        shb[i] = val;
    }
}

void f_silu_elementwise_mul_w3(RunState &s, int hidden_dim) {
    f_silu_elementwise_mul_w3_kernel<<<CEIL_DIV(hidden_dim, num_threads_small), num_threads_small>>>(s.hb, s.hb2,
                                                                                                  hidden_dim);
}

__global__ void accum_kernel(float *a, float *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        a[i] += b[i];
    }
}

void accum(float *a, float *b, int size) {
    accum_kernel<<<CEIL_DIV(size, num_threads_small), num_threads_small>>>(a, b, size);
}

float *forward(Transformer &transformer, int token, int pos) {
    // Convenience variables
    Config &p = transformer.config;
    TransformerWeights &w = transformer.weights;
    RunState &s = transformer.state;
    float *x = s.x;
    int dim = p.dim;
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    int kv_mul = p.n_heads / p.n_kv_heads; // Integer multiplier of the kv sharing in multiquery
    int hidden_dim = p.hidden_dim;
    int head_size = dim / p.n_heads;

    // Copy the token embedding into x
    float *content_row = w.token_embedding + token * dim;
    CUDA_CHECK(cudaMemcpy(x, content_row, dim * sizeof(*x), cudaMemcpyDeviceToDevice));

    // Forward all the layers
    for (unsigned long long l = 0; l < p.n_layers; l++) {
        // Attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight + l * dim, dim);

        // Key and value point to the kv cache
        int loff = l * p.max_seq_len * kv_dim; // KV cache layer offset
        s.k = s.key_cache + loff + pos * kv_dim;
        s.v = s.value_cache + loff + pos * kv_dim;

        // Calculate QKV matrixes with matmuls between weight matrixes and inputs 
        matmul(s.q, s.xb, w.wq + l * dim * dim, dim, dim);
        matmul(s.k, s.xb, w.wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s.v, s.xb, w.wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE relative positional encoding
        RoPe_rotation(pos, s, dim, kv_dim, head_size);

        // Multihead attention
        multi_head_attention(pos, p, s, kv_dim, kv_mul, head_size, loff);

        // Final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim);

        // Residual connection back into x
        accum(x, s.xb2, dim);

        // FFN rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight + l * dim, dim);

        // FFN computation
        matmul(s.hb, s.xb, w.w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        f_silu_elementwise_mul_w3(s, hidden_dim);

        // Final matmul to get the output of the FFN
        matmul(s.xb, s.hb, w.w2 + l * hidden_dim * dim, hidden_dim, dim);

        // Residual connection
        accum(x, s.xb, dim);
    }

    // Final rmsnorm
    rmsnorm(x, x, w.rms_final_weight, dim);

    // Classifier into logits
    matmul(s.logits_gpu, x, w.wcls, p.dim, p.vocab_size);
    CUDA_CHECK(cudaMemcpy(s.logits, s.logits_gpu, p.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return s.logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
// ----------------------------------------------------------------------------
struct TokenIndex {
    const char *str;
    int id;
};

struct Tokenizer {
    std::vector<char *> vocab;
    std::vector<float> vocab_scores;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // Stores all single-byte strings

    Tokenizer() : vocab_size(0), max_token_length(0) {
        memset(byte_pieces, 0, sizeof(byte_pieces));
    }
};

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *) a)->str, ((TokenIndex *) b)->str);
}

void build_tokenizer(Tokenizer &t, const char *tokenizer_path, int vocab_size) {
    t.vocab_size = vocab_size;
    t.vocab.resize(vocab_size);
    t.vocab_scores.resize(vocab_size);
    for (int i = 0; i < 256; i++) {
        t.byte_pieces[i * 2] = static_cast<unsigned char>(i);
        t.byte_pieces[i * 2 + 1] = '\0';
    }
    // Read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        printf("Error opening file: %s\n", strerror(errno));
        fprintf(stderr, "Couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t.max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Failed to read max_token_length\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(&t.vocab_scores[i], sizeof(float), 1, file) != 1) {
            fprintf(stderr, "Failed to read vocab_scores\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "Failed to read token length\n");
            exit(EXIT_FAILURE);
        }
        t.vocab[i] = new char[len + 1];
        if (fread(t.vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "Failed to read token string\n");
            exit(EXIT_FAILURE);
        }
        t.vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer &t) {
    for (auto &str : t.vocab) {
        delete[] str;
    }
    t.vocab.clear();
    t.vocab_scores.clear();
    t.sorted_vocab.clear();
}

const char *decode(Tokenizer &t, int prev_token, int token) {
    char *piece = t.vocab[token];
    // Following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if (prev_token == 1 && piece[0] == ' ') {
        piece++;
    }
    // Handle raw bytes
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = reinterpret_cast<char *>(t.byte_pieces) + byte_val * 2;
    }
    return piece;
}

void safe_printf(const char *piece) {
    // Skip non-printable bytes
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(const char *str, std::vector<TokenIndex> &sorted_vocab) {
    // Efficiently find the perfect match for str in vocab
    TokenIndex tok = {str, 0};
    auto it = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), tok,
                               [](const TokenIndex &a, const TokenIndex &b) {
                                   return strcmp(a.str, b.str) < 0;
                               });
    if (it != sorted_vocab.end() && strcmp(it->str, str) == 0) {
        return it->id;
    }
    return -1;
}

void encode(Tokenizer &t, const char *text, int8_t bos, int8_t eos, std::vector<int> &tokens) {
    // Encode the string text into tokens
    if (text == nullptr) {
        fprintf(stderr, "Cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t.sorted_vocab.empty()) {
        // Lazily sort the vocabulary
        for (int i = 0; i < t.vocab_size; i++) {
            t.sorted_vocab.push_back({t.vocab[i], i});
        }
        std::sort(t.sorted_vocab.begin(), t.sorted_vocab.end(),
                  [](const TokenIndex &a, const TokenIndex &b) {
                      return strcmp(a.str, b.str) < 0;
                  });
    }

    // Temporary buffer for merge candidates
    char *str_buffer = new char[t.max_token_length * 2 + 3]; // +3 for null terminator and possible UTF-8 bytes
    size_t str_len = 0;

    // Add optional BOS token
    if (bos) tokens.push_back(1);

    // Add dummy prefix if text is not empty
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t.sorted_vocab);
        tokens.push_back(dummy_prefix);
    }

    // Process the raw UTF-8 byte sequence
    const char *c = text;
    while (*c != '\0') {
        if ((*c & 0xC0) != 0x80) {
            // New codepoint
            str_len = 0;
        }
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        // If next character is not a continuation byte or buffer is full
        if ((*(c + 1) & 0xC0) != 0x80 || str_len >= 4) {
            int id = str_lookup(str_buffer, t.sorted_vocab);
            if (id != -1) {
                tokens.push_back(id);
            } else {
                // Byte fallback encoding
                for (size_t i = 0; i < str_len; i++) {
                    tokens.push_back(static_cast<unsigned char>(str_buffer[i]) + 3);
                }
            }
            str_len = 0;
        }
        c++;
    }

    // Merge the best consecutive pair each iteration
    while (true) {
        float best_score = -1e10f;
        int best_id = -1;
        int best_idx = -1;

        for (size_t i = 0; i < tokens.size() - 1; i++) {
            // Check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t.vocab[tokens[i]], t.vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t.sorted_vocab);
            if (id != -1 && t.vocab_scores[id] > best_score) {
                best_score = t.vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break;
        }

        // Merge the consecutive pair
        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    // Add optional EOS token
    if (eos) tokens.push_back(2);

    delete[] str_buffer;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// Sampling can be done in one way: greedy argmax
// ----------------------------------------------------------------------------
int sample_argmax(const float *probabilities, int n) {
    // Return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

// ----------------------------------------------------------------------------
// Utilities: time
// ----------------------------------------------------------------------------
long time_in_ms() {
    // Return time in milliseconds
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// Generation loop
// ----------------------------------------------------------------------------
void generate(Transformer &transformer, Tokenizer &tokenizer, const char *prompt, int max_new_tokens) {
    if (prompt == nullptr) { prompt = ""; }

    // Encode the prompt into tokens
    std::vector<int> prompt_tokens;
    encode(tokenizer, prompt, 1, 0, prompt_tokens);
    if (prompt_tokens.empty()) {
        fprintf(stderr, "Expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // Start the main loop
    long start = 0;  // Used to time the code
    int next;        // Next token in the sequence
    int token = prompt_tokens[0]; // Start with the first token in the prompt
    int pos = 0;     // Position in the sequence
    while (pos < max_new_tokens - 1) {
        // Forward the transformer to get logits for the next token
        float *logits = forward(transformer, token, pos);

        // Advance the state machine
        if (pos < static_cast<int>(prompt_tokens.size()) - 1) {
            // If processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_argmax(logits, transformer.config.vocab_size);
        }
        pos++;

        // Terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // Print the token as string
        const char *piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        // Initialize the timer after the first iteration
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // Report achieved tokens per second
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "Token count: %d, elapsed: %fs, %d tokens/s\n",
                pos + 1, (end - start) / 1000.0f, static_cast<int>((pos - 1) / ((end - start) / 1000.0f)));
    }
}

// ----------------------------------------------------------------------------
// Main function
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    // Default parameters
    const char *checkpoint_path = "stories15M.bin";  // e.g. out/model.bin
    const char *tokenizer_path = "tokenizer.bin";
    int max_new_tokens = 1000;                         // Number of max_new_tokens to run for
    const char *prompt = "Test";           // Prompt string

    if (argc >= 2) { prompt = argv[1]; }

    // Build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(transformer, checkpoint_path);
    if (max_new_tokens > transformer.config.max_seq_len)
        max_new_tokens = transformer.config.max_seq_len; // Override to max length

    // Build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(tokenizer, tokenizer_path, transformer.config.vocab_size);

    // Run!
    generate(transformer, tokenizer, prompt, max_new_tokens);

    // Memory and file handles cleanup
    free_tokenizer(tokenizer);
    free_transformer(transformer);
    return 0;
}
