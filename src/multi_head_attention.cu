#include "multi_head_attention.cuh"


namespace llama {

__global__ void multi_head_attention_kernel(int pos, int seq_len, float *q, float *k, float *v, float *att, float *out,
                                            int dim, int kv_dim, int kv_mul, int head_size) {
    int h = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (t > pos) return;

    __shared__ float s_q[64];  // Assuming max head_size is 64
    
    // Load query into shared memory
    if (threadIdx.x < head_size) {
        s_q[threadIdx.x] = q[h * head_size + threadIdx.x];
    }
    __syncthreads();

    // Calculate attention score
    float score = 0.0f;
    for (int i = 0; i < head_size; i++) {
        score += s_q[i] * k[(h / kv_mul) * head_size + t * kv_dim + i];
    }
    score /= sqrtf(head_size);
    
    // Store score in global memory
    att[h * seq_len + t] = score;
    __syncthreads();

    // Softmax is handled separately

    // Weighted sum of values
    for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
        float val = 0.0f;
        for (int j = 0; j <= pos; j++) {
            val += att[h * seq_len + j] * v[(h / kv_mul) * head_size + j * kv_dim + i];
        }
        out[h * head_size + i] = val;
    }
}

void multi_head_attention(int pos, int seq_len, float *q, float *k, float *v, float *att, float *out,
                          int dim, int kv_dim, int n_heads, int kv_mul, int head_size) {
    dim3 grid(1, n_heads);
    dim3 block(256);
    multi_head_attention_kernel<<<grid, block>>>(pos, seq_len, q, k, v, att, out, dim, kv_dim, kv_mul, head_size);
}

    
}