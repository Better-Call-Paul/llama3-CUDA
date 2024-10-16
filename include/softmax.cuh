#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace llama {

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

// warp reduce max and sum
template<typename T>
__device__ void warp_reduce_sum(T& sum) {
    // each iteration gets value of val at lane X + offset 
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        // __shfl_down_sync uses tree reduction to compute the sum for eahc val in each thread in a warp
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
}

template<typename T>
__device__ void warp_reduce_max(T& max_val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(FULL_MASK, max_val, offset));
    }
}

// block reduce max and sum
template<typename T>
__device__ void block_reduce_sum(T sum) {

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // thread within warp index 
    static __shared__ shmem[32]; // holds partial sums
    
    warp_reduce_sum(sum);

    // final sum will be in the first thread
    if (lane_id == 0) shmem[warp_id] = sum;

    __syncthreads();

    if (warp_id == 0) {
        // blockDim.x / WARP_SIZE == number of active warps
        // must be within that range to contribbute its partial sum
        sum = (lane_id < (blockDim.x / WARP_SIZE) ? shmem[lane_id] : 0);
        
        // first warp does final reduction on partial sums to compute final block sum
        warp_reduce_sum(sum);
    }
}

template<typename T>
__device__ void block_reduce_max(T max_val) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    static __shared__ shmem[32];

    warp_reduce_max(max_val);

    if (lane_id == 0) shmem[warp_id] = max_val

    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane_id < (blockDim.x / WARP_SIZE) ? shmem[lane_id] : 0);

        warp_reduce_max(max_val);
    }
}

/*
 * M x K Matrix
 */
template<typename T>
__global__ void softmax(const uint M, const uint K, const T* __restrict__ input, T* __restrict__ output) {

    const uint row = threadIdx.x;

    if (row >= M) return;

    // Calculate the starting index for this row
    const T* row_input = input + row * num_cols;
    T* row_output = output + row * num_cols;

    // Initialize thread index
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // Shared memory allocation for reductions
    extern __shared__ float shared_mem[]; // Adjust type if necessary

    // Find the maximum value in the row for numerical stability
    // Each thread processes multiple elements if necessary
    float max_val = -FLT_MAX;
    for (int i = tid; i < num_cols; i += total_threads) {
        float val = static_cast<float>(row_input[i]);
        max_val = max(max_val, val);
    }

    // Reduce to find the global max
    max_val = block_reduce_max(max_val);

    // Subtract the max and compute exponentials
    float sum_exp = 0.0f;
    for (int i = tid; i < num_cols; i += total_threads) {
        float val = static_cast<float>(row_input[i]);
        float shifted = val - max_val;
        float exp_val = expf(shifted);
        row_output[i] = exp_val; // Temporarily store exponentials
        sum_exp += exp_val;
    }

    // Compute the sum of exponentials
    sum_exp = block_reduce_sum(sum_exp);

    // Normalize to get Softmax probabilities
    for (int i = tid; i < num_cols; i += total_threads) {
        row_output[i] /= sum_exp;
    }

}

}  

#endif 