#include "rope_rotation.cuh"


namespace llama {

__global__ void rope_rotation_kernel(int pos, float *q, float *k, int dim, int kv_dim, int head_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim / 2) return;

    int head_dim = idx % (head_size / 2);
    float freq = 1.0f / powf(10000.0f, head_dim / (float)(head_size / 2));
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);

    int rotn = idx < kv_dim / 2 ? 2 : 1;
    for (int v = 0; v < rotn; v++) {
        float *vec = v == 0 ? q : k;
        float v0 = vec[idx * 2];
        float v1 = vec[idx * 2 + 1];
        vec[idx * 2] = v0 * fcr - v1 * fci;
        vec[idx * 2 + 1] = v0 * fci + v1 * fcr;
    }
}

void rope_rotation(int pos, float *q, float *k, int dim, int kv_dim, int head_size) {
    int block_size = 256;
    int grid_size = (dim / 2 + block_size - 1) / block_size;
    rope_rotation_kernel<<<grid_size, block_size>>>(pos, q, k, dim, kv_dim, head_size);
}
}