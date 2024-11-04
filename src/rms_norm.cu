#include "rms_norm.cuh"

namespace llama {

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size, int elementsPerThread) {
    // Parallel reduction of sum of squares via CUB
    float ss = 0.0f;
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size)
            ss += x[j] * x[j];
    }
    using BlockReduce = cub::BlockReduce<float, num_threads_large>;
    __shared__ typename BlockReduce::TempStorage temp;
    ss = BlockReduce(temp).Sum(ss);

    // Calculate normalization factor
    __shared__ float shared_ss;
    if (threadIdx.x == 0) {
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        shared_ss = ss;
    }
    __syncthreads();
    ss = shared_ss;

    // Normalize and scale
    for (int i = 0; i < elementsPerThread; i++) {
        int j = threadIdx.x + i * num_threads_large;
        if (j < size) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}

void rmsnorm(float *o, float *x, float *weight, int size) {
    int elementsPerThread = CEIL_DIV(size, num_threads_large);
    rmsnorm_kernel<<<1, num_threads_large>>>(o, x, weight, size, elementsPerThread);
}

    
}