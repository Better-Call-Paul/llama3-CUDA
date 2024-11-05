#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>

#define CEIL_DIV(M, N) ((M + N - 1) / N)

/*
 * A: M x K
 * B: K x N
 * C: M x N
 */
template<int BM, int BN, int BK, int TM, int TN>
__global__ void wmma_gemm(int M, int N, int K, const float alpha, const float *A, const float *B, const float beta, float *C) {

    // move ptrs into curr pos
    // load data into shmem
    // go over each tile in shmem
    // wmma sync
    // store back to gmem
        


    


}