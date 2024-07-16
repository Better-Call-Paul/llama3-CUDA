#ifndef MULTI_HEAD_ATTENTION_CUH
#define MULTI_HEAD_ATTENTION_CUH

namespace llama {
    
__global__ void multi_head_attention_kernel(int pos, int seq_len, float *q, float *k, float *v, float *att, float *out,
                                            int dim, int kv_dim, int kv_mul, int head_size);

void multi_head_attention(int pos, int seq_len, float *q, float *k, float *v, float *att, float *out,
                          int dim, int kv_dim, int n_heads, int kv_mul, int head_size);
}

#endif 