#include "mat_mul.cuh"
#include <string>
#include <stdexcept>
#include <iostream>

namespace llama {
    
 template<typename T, int BLOCK_SIZE, int TILE_SIZE>
 __global__ void mat_mul_kernel(const T* __restrict__ inputv, const T* __restrict__ weightv, T* __restrict__ outputv, int n, int d) {
    // shared mem
    __shared__ T shared_mem[TILE_SIZE];

    T sum = 0.0f;
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < n; i += TILE_SIZE) {
        if (i + threadIdx.x < n) {
            shared_mem[threadIdx.x] = inputv[threadIdx.x + i];
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_SIZE && i + j < n; j++) {
            sum += shared_mem[j] * weightv[row * n + (i + j)];
        }
        __syncthreads();
    }
    if (row < d) {
        outputv[row] = sum;
    }
 }

 __global__ void silu_elementwise_mul_kernel(float *a, float *b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = a[idx];
        // SiLU activation: x * sigmoid(x)
        val *= (1.0f / (1.0f + expf(-val)));
        // Elementwise multiplication
        a[idx] = val * b[idx];
    }
}

void silu_elementwise_mul(float *a, float *b, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    silu_elementwise_mul_kernel<<<grid_size, block_size>>>(a, b, size);
}

__global__ void matmul_kernel(float *outpt, const float *input, const float *weight, int n, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += weight[row * n + i] * input[i];
        }
        outpt[row] = sum;
    }
}1

void matmul(float *outpt, const float *input, const float *weight, int n, int d) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (d + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(outpt, input, weight, n, d);
    
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("matmul kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaDeviceSynchronize returned error after launching matmul kernel: " 
                                 + std::string(cudaGetErrorString(cudaStatus)));
    }
}

template<const uint block_size_M, const uint block_size_N, const uint block_size_K, const uint tile_size_M, const uint tile_size_N>
__global__ void mixed_precision_gemm(int M, int N, int K, const float alpha, const __half *A, const __half *B, const float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    __shared__ __half A_Shmem[block_size_M * block_size_K];
    __shared__ __half B_Shmem[block_size_K * block_size_N];


    // block size N * tile size N is span of one column
    // represent the thread's position within the tile 
    // maps thread to its exact tile computation
    int threadRow = threadIdx.x / (block_size_N * tile_size_N);
    int threadCol = threadIdx.x % (block_size_N * tile_size_N);

    A += cRow * block_size_M * K;
    B += cCol * block_size_N;
    C += cRow * block_size_M * N + cCol * block_size_N;


    // calculate the indices that the thread will load into SHMEM
    // load 128 bit / 32 bit = 4 bits per thread at each step
    const uint innerRowA = threadIdx.x / (block_size_K / 4);
    const uint innerColA = threadIdx.x % (block_size_K / 4);
    const uint innerRowB = threadIdx.x / (block_size_N / 4);
    const uint innerColB = threadIdx.x % (block_size_N / 4);

    // allocate cache local to thread 
    float threadResults[tile_size_M * tile_size_N] = {0.0f};
    float register_M[tile_size_M] = {0.0f};
    float register_N[tile_size_N] = {0.0f};

    // loop over all block tiles 
    for (int bckIdx = 0; bckIdx < K; bckIdx += block_size_K) [
        // populate shmem cache
        fl
    ]




}

template <const int BM, const int BN, const int BK>
__global__ void hgemm_wmma(int M, int N, int K, __half alpha, __half *A,
                           __half *B, __half beta, __half *C) {
    // WMMA tile dimensions
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Calculate global warp indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    // Each warp computes one WMMA tile
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) return;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension tiles
    for (int k = 0; k < K; k += WMMA_K) {
        // Load the inputs
        if (warpM * WMMA_M < M && (k + WMMA_K) <= K) {
            int aRow = warpM * WMMA_M;
            int aCol = k;
            const __half *A_tile = &A[aRow * K + aCol];
            wmma::load_matrix_sync(a_frag, A_tile, K);
        }

        if (warpN * WMMA_N < N && (k + WMMA_K) <= K) {
            int bRow = k;
            int bCol = warpN * WMMA_N;
            const __half *B_tile = &B[bRow * N + bCol];
            wmma::load_matrix_sync(b_frag, B_tile, N);
        }

        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Load C matrix tile
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    float *C_tile = reinterpret_cast<float *>(&C[cRow * N + cCol]);

    // Scale the result by alpha and beta
    // Note: WMMA accumulates in FP32
    if (cRow < M && cCol < N) {
        // Load the current value of C (if beta != 0)
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_orig_frag;
        if (beta != __float2half(0.0f)) {
            // Convert __half *C to float *C for accumulation
            float C_vals[WMMA_M * WMMA_N];
            for (int i = 0; i < WMMA_M; i++) {
                for (int j = 0; j < WMMA_N; j++) {
                    int idx = (cRow + i) * N + (cCol + j);
                    if ((cRow + i) < M && (cCol + j) < N) {
                        C_vals[i * WMMA_N + j] = __half2float(C[idx]);
                    } else {
                        C_vals[i * WMMA_N + j] = 0.0f;
                    }
                }
            }
            wmma::load_matrix_sync(c_orig_frag, C_vals, N, wmma::mem_row_major);
        } else {
            wmma::fill_fragment(c_orig_frag, 0.0f);
        }

        // Compute alpha * (A * B) + beta * C
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = __half2float(alpha) * c_frag.x[i] + __half2float(beta) * c_orig_frag.x[i];
        }

        // Store the result
        __half C_result[WMMA_M * WMMA_N];
        for (int i = 0; i < c_frag.num_elements; i++) {
            C_result[i] = __float2half(c_frag.x[i]);
        }
        wmma::store_matrix_sync(C_tile, c_frag, N, wmma::mem_row_major);
    }
}





}