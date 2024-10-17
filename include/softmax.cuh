#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace llama {

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

// warp reduce max and sum
__device__ float warp_reduce_sum(float sum);

__device__ float warp_reduce_max(float max_val);

// block reduce max and sum
__device__ float block_reduce_sum(float sum, float *shmem);

__device__ float block_reduce_max(float max_val, float *shmem);

/*
 * M x K Matrix
 */
__global__ void softmax(const uint M, const uint K, const __half* __restrict__ input, float* __restrict__ output);

}  

#endif 