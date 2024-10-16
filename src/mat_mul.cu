#include "mat_mul.cuh"
#include <string>
#include <stdexcept>
#include <iostream>

namespace llama {
    
template<const uint block_size_M, const uint block_size_N, const uint block_size_K, const uint tile_size_M, const uint tile_size_N>
__global__ void half_gemm(int N, int M, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadRow = threadIdx.x / block_size_N;
    const uint threadCol = threadIdx.x % block_size_N;

    A += cRow * block_size_M * K;
    B += cCol * block_size_N;
    C += cRow * block_size_M * N + cCol * block_size_N;

    // warp level gmem coalescing indices
    const uint innerRowA = threadIdx.x / block_size_M;
    const uint innerColA = threadIdx.x % block_size_M;
    const uint innerRowB = threadIdx.x / block_size_N;
    const uint innerColB = threadIdx.x % block_size_N;

    __shared__ __half A_Shmem[block_size_M * block_size_K];
    __shared__ __half B_Shmem[block_size_K * block_size_N];

    float thread_results[tile_size_M] = {0.0f};
    
    for (int bckIdx = 0; bckIdx < K; bckIdx += block_size_K) {
        
        A_Shmem[innerRowA * block_size_K + innerColA] = A[innerRowA * K + innerColA];
        B_Shmem[innerRowB * block_size_N + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += block_size_N;
        B += block_size_N * N;

        for (uint dotIdx = 0; dotIdx < block_size_K; ++dotIdx) {
            float temp = B_Shmem[dotIdx * block_size_N + threadCol];

            for (uint resIdx = 0; resIdx < tile_size_M; ++resIdx) {
                thread_results[resIdx] += A_Shmem[(threadRow * tile_size_M + resIdx) * block_size_K + threadCol] * temp;
            }
        } 

        __syncthreads();
     
    }
    // load into c 
    for (uint resIdx = 0; resIdx < tile_size_M; ++resIdx) {
        C[(threadRow * tile_size_M + resIdx) * N + threadCol] = alpha * thread_results[resIdx] + beta * C[(threadRow * tile_size_M + resIdx) * N + threadCol];
    }

}


}