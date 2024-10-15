#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#include "utils.cuh"

namespace llama {

const int BLOCK_SIZE = 32;

template<typename T, int BLOCK_SIZE, int TILE_SIZE>
__global__ void mat_mul_kernel(const T* __restrict__ inputv, const T* __restrict__ weightv, T* __restrict__ outputv, int n, int d);

__global__ void silu_elementwise_mul_kernel(float *a, float *b, int size);

void silu_elementwise_mul(float *a, float *b, int size);

__global__ void matmul_kernel(float *outpt, const float *input, const float *weight, int n, int d);

void matmul(float *outpt, const float *input, const float *weight, int n, int d);

template<const uint block_size_M, const uint block_size_N, const uint block_size_K, const uint tile_size_M, const uint tile_size_N>
__global__ void mixed_precision_gemm(int M, int N, int K, const float alpha, const __half *A, const __half *B, const float beta, float *C);

}

#endif