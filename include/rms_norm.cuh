#ifndef RMS_NORM_CUH
#define RMS_NORM_CUH

namespace llama {
    
__global__ void rmsnorm_kernel(float *o, float *x, float *weight, int size);

void rmsnorm(float *o, float *x, float *weight, int size);

}

#endif