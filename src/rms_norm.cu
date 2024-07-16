#include "rms_norm.cuh"

namespace llama {

__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size) {
    __shared__ float sum;
    sum = 0.0f;
    __syncthreads();

    // Calculate sum of squares
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = x[i];
        thread_sum += val * val;
    }
    atomicAdd(&sum, thread_sum);
    __syncthreads();

    // Normalize and scale
    float inv_sqrt = rsqrtf(sum / size + 1e-5f);
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        o[i] = weight[i] * (inv_sqrt * x[i]);
    }
}

void rmsnorm(float *o, float *x, float *weight, int size) {
    rmsnorm_kernel<<<1, 256>>>(o, x, weight, size);
}
    
}