#ifndef ROPE_ROTATION_CUH
#define ROPE_ROTATION_CUH

namespace llama {

__global__ void rope_rotation_kernel(int pos, float *q, float *k, int dim, int kv_dim, int head_size);

void rope_rotation(int pos, float *q, float *k, int dim, int kv_dim, int head_size);

}

#endif