//#include "softmax.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace llama {

// Define constants
#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32

// Warp reduce sum
__device__ float warp_reduce_sum(float sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    return sum;
}

// Warp reduce max
__device__ float warp_reduce_max(float max_val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(FULL_MASK, max_val, offset));
    }
    return max_val;
}

// Block reduce sum
__device__ float block_reduce_sum(float sum, float* shmem) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // thread within warp index 
    
    // Reduce within warp
    sum = warp_reduce_sum(sum);
    
    // Write reduced sum to shared memory
    if (lane_id == 0) shmem[warp_id] = sum;
    
    __syncthreads();
    
    // Final reduce within first warp
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / WARP_SIZE) ? shmem[lane_id] : 0.0f);
        sum = warp_reduce_sum(sum);
    }
    
    __syncthreads();
    
    return sum;
}

// Block reduce max
__device__ float block_reduce_max(float max_val, float* shmem) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Reduce within warp
    max_val = warp_reduce_max(max_val);
    
    // Write reduced max to shared memory
    if (lane_id == 0) shmem[warp_id] = max_val;
    
    __syncthreads();
    
    // Final reduce within first warp
    if (warp_id == 0) {
        max_val = (lane_id < (blockDim.x / WARP_SIZE) ? shmem[lane_id] : -FLT_MAX);
        max_val = warp_reduce_max(max_val);
    }
    
    __syncthreads();
    
    return max_val;
}

/*
 * M x K Matrix Softmax Kernel
 */
__global__ void softmax(const uint M, const uint K, const __half* __restrict__ input, float* __restrict__ output) {
    
    extern __shared__ float shmem[];

    const uint row = blockIdx.x;
    // Each row computes one block

    // Move pointers to the current block's row
    input += row * K;
    output += row * K;

    int tid = threadIdx.x;
    int total_threads = blockDim.x; // stride for better GMEM coalescing 

    float max_val = -FLT_MAX;

    // Compute partial maxes
    for (int i = tid; i < K; i += total_threads) {
        float val = __half2float(input[i]);
        max_val = fmaxf(max_val, val);
    }

    // Find global max using block reduction
    max_val = block_reduce_max(max_val, shmem);
    
    float sum_exponents = 0.0f;
    // Compute partial sums of exponentials
    for (int i = tid; i < K; i += total_threads) {
        float curr_val = __half2float(input[i]);
        float exponent = expf(curr_val - max_val);
        sum_exponents += exponent;
        output[i] = exponent; 
    }

    // Find the sum of exponentials using block reduction
    sum_exponents = block_reduce_sum(sum_exponents, shmem);
    
    __syncthreads(); // Ensure all exponentials are computed before normalization

    // Normalize the exponentials to get softmax probabilities
    for (int i = tid; i < K; i += total_threads) {
        output[i] /= sum_exponents;
    }

}

} // namespace llama
