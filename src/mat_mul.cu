#include "mat_mul.cuh"
#include <string>
#include <stdexcept>
#include <iostream>

namespace llama {
    
 template<typename T, int BLOCK_SIZE, int TILE_SIZE>
 __global__ void mat_mul_kernel(const T* __restrict__ inputv, const T* __restrict__ weightv, T* __restrict__ outputv, int n, int d) {
    // shared mem
    __shared__ T shared_mem[TILE_SIZE];

    T sum = 0.0f;
    int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < n; i += TILE_SIZE) {
        if (i + threadIdx.x < n) {
            shared_mem[threadIdx.x] = inputv[threadIdx.x + i];
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < TILE_SIZE && i + j < n; j++) {
            sum += shared_mem[j] * weightv[row * n + (i + j)];
        }
        __syncthreads();
    }
    if (row < d) {
        outputv[row] = sum;
    }
 }

 __global__ void silu_elementwise_mul_kernel(float *a, float *b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = a[idx];
        // SiLU activation: x * sigmoid(x)
        val *= (1.0f / (1.0f + expf(-val)));
        // Elementwise multiplication
        a[idx] = val * b[idx];
    }
}

void silu_elementwise_mul(float *a, float *b, int size) {
    int block_size = 256;
    int grid_size = (size + block_size - 1) / block_size;
    silu_elementwise_mul_kernel<<<grid_size, block_size>>>(a, b, size);
}

__global__ void matmul_kernel(float *outpt, const float *input, const float *weight, int n, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += weight[row * n + i] * input[i];
        }
        outpt[row] = sum;
    }
}

void matmul(float *outpt, const float *input, const float *weight, int n, int d) {
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (d + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(outpt, input, weight, n, d);
    
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("matmul kernel launch failed: " + std::string(cudaGetErrorString(cudaStatus)));
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("cudaDeviceSynchronize returned error after launching matmul kernel: " 
                                 + std::string(cudaGetErrorString(cudaStatus)));
    }
}

}