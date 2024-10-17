// test_softmax.cu

#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cfloat>

// Error checking macros
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " - " << cudaGetErrorString(err) << std::endl;       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CHECK_CUDNN(call)                                                     \
    do {                                                                      \
        cudnnStatus_t status = call;                                          \
        if (status != CUDNN_STATUS_SUCCESS) {                                 \
            std::cerr << "cuDNN error in " << __FILE__ << ":" << __LINE__     \
                      << " - " << cudnnGetErrorString(status) << std::endl;  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// GpuTimer structure for CUDA event timing
struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
    float elapsedTime;

    GpuTimer() : elapsedTime(0.0f) {
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
    }

    ~GpuTimer() {
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    void Start() {
        CHECK_CUDA(cudaEventRecord(start, 0));
    }

    void Stop() {
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    }
};

// Namespace for custom implementations
namespace llama {

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

// Warp reduce sum
template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

// Warp reduce max
template<typename T>
__device__ T warp_reduce_max(T val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val;
}

// Block reduce sum
template<typename T>
__device__ T block_reduce_sum(T val) {
    __shared__ T shared_sum[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared_sum[warp_id] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    if (warp_id == 0) {
        val = (lane < (blockDim.x / WARP_SIZE)) ? shared_sum[lane] : (T)0;
        val = warp_reduce_sum(val);
    }
    return val;
}

// Block reduce max
template<typename T>
__device__ T block_reduce_max(T val) {
    __shared__ T shared_max[WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) {
        shared_max[warp_id] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    if (warp_id == 0) {
        val = (lane < (blockDim.x / WARP_SIZE)) ? shared_max[lane] : (T)-FLT_MAX;
        val = warp_reduce_max(val);
    }
    return val;
}

/*
 * M x K Matrix Softmax Kernel
 */
template<typename T>
__global__ void softmax_kernel(const uint M, const uint K, const T* __restrict__ input, T* __restrict__ output) {
    // Each block processes one row
    extern __shared__ float shared_mem[]; // Shared memory for reductions
    int row = blockIdx.x;

    if (row >= M) return;

    const T* row_input = input + row * K;
    T* row_output = output + row * K;

    // First, find the maximum value in the row for numerical stability
    float max_val = -FLT_MAX;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        max_val = fmaxf(max_val, val);
    }

    // Reduce to find the global max
    max_val = block_reduce_max(max_val);

    // Synchronize to ensure max_val is computed
    __syncthreads();

    // Subtract max and compute exponentials, then sum them
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]);
        float shifted = val - max_val;
        float exp_val = expf(shifted);
        row_output[i] = static_cast<T>(exp_val); // Temporarily store exponentials
        sum_exp += exp_val;
    }

    // Reduce to find the sum of exponentials
    sum_exp = block_reduce_sum(sum_exp);

    // Synchronize to ensure sum_exp is computed
    __syncthreads();

    // Normalize to get Softmax probabilities
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        row_output[i] = static_cast<T>(static_cast<float>(row_output[i]) / sum_exp);
    }
}

} // namespace llama

int main() {
    // Parameters
    const uint M = 128;          // Number of rows (batch size)
    const uint K = 1000;         // Number of classes
    const size_t size = M * K;
    const size_t bytes = size * sizeof(float);

    // Host vectors
    std::vector<float> h_input(size);
    std::vector<float> h_output_custom(size);
    std::vector<float> h_output_cudnn(size);

    // Initialize input with random data
    for(auto &val : h_input) {
        val = static_cast<float>(rand()) / RAND_MAX;
    }

    // Device pointers
    float *d_input, *d_output_custom, *d_output_cudnn;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output_custom, bytes));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, bytes));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create tensor descriptor
    cudnnTensorDescriptor_t tensor_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&tensor_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(tensor_desc,
                                          /*format=*/CUDNN_TENSOR_NCHW,
                                          /*dataType=*/CUDNN_DATA_FLOAT,
                                          /*batch_size=*/M,
                                          /*channels=*/1,
                                          /*height=*/1,
                                          /*width=*/K));

    // Set Softmax parameters
    cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE;
    cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_INSTANCE;

    // Define alpha and beta
    float alpha = 1.0f, beta = 0.0f;

    // Initialize timer
    GpuTimer timer;

    // Define CUDA kernel launch parameters
    int threads_per_block = 256;
    int blocks = M; // One block per row
    size_t shared_mem_size = threads_per_block * sizeof(float); // Adjust as needed

    // Warm-up runs to stabilize GPU
    // Custom Softmax
    llama::softmax_kernel<float><<<blocks, threads_per_block, shared_mem_size>>>(M, K, d_input, d_output_custom);
    CHECK_CUDA(cudaDeviceSynchronize());

    // cuDNN Softmax
    CHECK_CUDNN(cudnnSoftmaxForward(cudnn,
                                    algo,
                                    mode,
                                    &alpha,
                                    tensor_desc,
                                    d_input,
                                    &beta,
                                    tensor_desc,
                                    d_output_cudnn));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark Custom Softmax
    int iterations = 100;
    float total_time_custom = 0.0f;

    timer.Start();
    for(int i = 0; i < iterations; ++i) {
        llama::softmax_kernel<float><<<blocks, threads_per_block, shared_mem_size>>>(M, K, d_input, d_output_custom);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.Stop();
    total_time_custom = timer.elapsedTime / iterations;

    std::cout << "Custom Softmax Time (per iteration): " << total_time_custom << " ms" << std::endl;

    // Benchmark cuDNN Softmax
    float total_time_cudnn = 0.0f;

    timer.Start();
    for(int i = 0; i < iterations; ++i) {
        CHECK_CUDNN(cudnnSoftmaxForward(cudnn,
                                        algo,
                                        mode,
                                        &alpha,
                                        tensor_desc,
                                        d_input,
                                        &beta,
                                        tensor_desc,
                                        d_output_cudnn));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    timer.Stop();
    total_time_cudnn = timer.elapsedTime / iterations;

    std::cout << "cuDNN Softmax Time (per iteration): " << total_time_cudnn << " ms" << std::endl;

    // Validate correctness
    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_output_custom.data(), d_output_custom, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_cudnn.data(), d_output_cudnn, bytes, cudaMemcpyDeviceToHost));

    // Compute the maximum absolute difference
    float max_diff = 0.0f;
    for(int i = 0; i < size; ++i) {
        float diff = fabs(h_output_custom[i] - h_output_cudnn[i]);
        if(diff > max_diff) max_diff = diff;
    }

    std::cout << "Max difference between custom and cuDNN Softmax: " << max_diff << std::endl;

    // Define a tolerance
    const float tolerance = 1e-5f;
    if(max_diff < tolerance) {
        std::cout << "Softmax correctness: PASS" << std::endl;
    } else {
        std::cout << "Softmax correctness: FAIL" << std::endl;
    }

    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(tensor_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output_custom));
    CHECK_CUDA(cudaFree(d_output_cudnn));

    return 0;
}
