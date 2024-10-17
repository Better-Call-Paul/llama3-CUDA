#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#include "utils.cuh"
#include <cuda_fp16.h>

namespace llama {

template<const uint block_size_M, const uint block_size_N, const uint block_size_K, const uint tile_size_M, const uint tile_size_N>
__global__ void half_gemm(int M, int N, int K, float alpha, __half *A, __half *B, float beta, float *C);


template<const uint block_size_M, const uint block_size_N, const uint block_size_K, const uint tile_size_M, const uint tile_size_N>
__global__ void wmma_gemm(int M, int N, int K, float alpha, const __half* A, const __half *B, float beta, float *C);

}

#endif