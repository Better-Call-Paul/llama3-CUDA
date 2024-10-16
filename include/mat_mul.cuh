#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

#include "utils.cuh"

namespace llama {

template<const uint block_size_M, const uint block_size_N, const uint block_size_K, const uint tile_size_M, const uint tile_size_N>
__global__ void half_gemm(int N, int M, int K, float alpha, const float *A, const float *B, float beta, float *C);

}

#endif