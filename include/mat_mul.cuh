#ifndef MAT_MUL_CUH
#define MAT_MUL_CUH

namespace llama {

template<typename T, int BLOCK_SIZE, int TILE_SIZE>
__global__ void mat_mul_kernel(const T* __restrict__ inputv, const T* __restrict__ weightv, T* __restrict__ outputv, int n, int d);


__global__ void silu_elementwise_mul_kernel(float *a, float *b, int size)

void silu_elementwise_mul(float *a, float *b, int size)

}

#endif