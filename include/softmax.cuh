#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace llama {

#define FULL_MASK 0xffffffff

template<typename T>
__device__ void warpReduceMax(T& val);

template<typename T>
__device__ void warpReduceSum(T& val);

template<typename T>
__device__ T blockReduceMax(T val);

template<typename T>
__device__ T blockReduceSum(T val);

template<typename T, int cols_per_thread>
__global__ void softmaxLocal(const T* input, T* output, size_t m, size_t n);

template<typename T, int block_size>
__global__ void softmaxLarge(const T* input, T* output, size_t m, const size_t n);

template<typename T>
__device__ T Inf();

}  

#endif 