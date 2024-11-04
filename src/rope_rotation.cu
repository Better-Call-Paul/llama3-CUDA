#include "rope_rotation.cuh"


namespace llama {

__global__ void RoPe_rotation_kernel(int pos, float *sq, float *sk, int kv_dim, int head_size) {
    int i = threadIdx.x * 2;
    if (i >= kv_dim)
        return;
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / static_cast<float>(head_size));
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        float *vec = v == 0 ? sq : sk; // The vector to rotate (query or key)
        float v0 = vec[i];
        float v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci;
        vec[i + 1] = v0 * fci + v1 * fcr;
    }
}

void RoPe_rotation(int pos, RunState &s, int dim, int kv_dim, int head_size) {
    RoPe_rotation_kernel<<<1, dim / 2>>>(pos, s.q, s.k, kv_dim, head_size);
}

}