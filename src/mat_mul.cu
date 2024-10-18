#include "mat_mul.cuh"
#include <string>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>
#include <mma.h>

using namespace nvcuda::mma;

namespace llama {

template<const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__global__ void wmma_gemm(int M, int N, int K, float alpha, const __half* A, const __half *B, float beta, float *C) {

    // get block positions
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // global position
    const uint globalRow = threadIdx.y * WMMA_M;
    const uint globalCol = threadIdx.x * WMMA_N;

    // move matrices to current block
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // initialize output fragments
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    extern __shared__ __half shared_mem[];
    __half* A_shared = shared_mem;
    __half* B_shared = shared_mem + BM * BK;


    // loop with stride for gmem coalescing
    for (int tileIdx = 0; tileIdx < CEIL_DIV(K, WMMA_K); tileIdx++) {
        
       // Calculate the starting indices for A and B
        int aTileRow = threadIdx.y * WMMA_M;
        int aTileCol = tileIdx * WMMA_K + threadIdx.x * WMMA_K;

        int bTileRow = tileIdx * WMMA_K + threadIdx.y * WMMA_K;
        int bTileCol = threadIdx.x * WMMA_N;

        // Load A tile into shared memory
        for (int i = 0; i < WMMA_M; ++i) {
            int a_row = globalRow + i;
            int a_col = aTileCol;
            if (a_row < M && a_col < K) {
                A_shared[(threadIdx.y * WMMA_M + i) * BK + threadIdx.x * WMMA_K] =
                    A[a_row * lda + a_col];
            } else {
                A_shared[(threadIdx.y * WMMA_M + i) * BK + threadIdx.x * WMMA_K] = __float2half(0.0f);
            }
        }

        // Load B tile into shared memory
        for (int i = 0; i < WMMA_K; ++i) {
            int b_row = bTileRow + i;
            int b_col = globalCol;
            if (b_row < K && b_col < N) {
                B_shared[(threadIdx.y * WMMA_K + i) * BN + threadIdx.x * WMMA_N] =
                    B[b_row * ldb + b_col];
            } else {
                B_shared[(threadIdx.y * WMMA_K + i) * BN + threadIdx.x * WMMA_N] = __float2half(0.0f);
            }
        }

        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;

        // load inputs into fragments
        // frag, mem input, col size
        load_matrix_sync(a_frag, A_shared, BK);
        load_matrix_sync(b_frag, B_shared, BN);

        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
        // dot prod
    }

    // Initialize C_frag with beta * C
    for (int i = 0; i < c_frag.num_elements; ++i) {
        int c_row = globalRow + (i / WMMA_N);
        int c_col = globalCol + (i % WMMA_N);
        if (c_row < M && c_col < N) {
            // Assuming row-major order for C
            c_frag.x[i] += beta * C_block[c_row * ldc + c_col];
        }
    }

    // Scale by alpha and store the result back to C_block
    for (int i = 0; i < WMMA_M; ++i) {
        for (int j = 0; j < WMMA_N; ++j) {
            int c_row = globalRow + i;
            int c_col = globalCol + j;
            if (c_row < M && c_col < N) {
                C_block[c_row * ldc + c_col] = alpha * c_frag.x[i * WMMA_N + j];
            }
        }
    }
    // load into c
}


    
template<const uint block_size_M, const uint block_size_N, const uint block_size_K, const uint tile_size_M, const uint tile_size_N>
__global__ void half_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

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