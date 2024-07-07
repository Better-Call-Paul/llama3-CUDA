#include "mat_mul.cuh"

namespace llama {
    
/**
 * @brief Performs matrix-vector multiplication on the GPU
 *
 * This kernel computes the product of a matrix w and a vector x, storing the result in xout.
 * It uses shared memory and tiling for improved performance.
 *
 * @tparam T The data type of the matrices and vectors (e.g., float, double, half)
 * @tparam BLOCK_SIZE The number of threads per block
 * @tparam TILE_SIZE The size of the tile (chunk) processed at a time
 *
 * @param[out] outpt Pointer to the output vector (d x 1)
 * @param[in] input Pointer to the input vector (n x 1)
 * @param[in] weight Pointer to the weight matrix (d x n)
 * @param[in] n The number of columns in weight
 * @param[in] d The number of rows in weight
 */
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

}