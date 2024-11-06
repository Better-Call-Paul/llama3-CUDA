#include "mat_mul.cuh"
#include <string>
#include <stdexcept>
#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>
#include <mma.h>

using namespace nvcuda::mma;

namespace llama {

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





}