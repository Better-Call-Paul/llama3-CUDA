
namespace llama {
/*
float *forward(Transformer &transformer, int token, int pos) {
    // Convenience variables
    Config &p = transformer.config;
    TransformerWeights &w = transformer.weights;
    RunState &s = transformer.state;
    float *x = s.x;
    int dim = p.dim;
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    int kv_mul = p.n_heads / p.n_kv_heads; // Integer multiplier of the kv sharing in multiquery
    int hidden_dim = p.hidden_dim;
    int head_size = dim / p.n_heads;

    // Copy the token embedding into x
    float *content_row = w.token_embedding + token * dim;
    CUDA_CHECK(cudaMemcpy(x, content_row, dim * sizeof(*x), cudaMemcpyDeviceToDevice));

    // Forward all the layers
    for (unsigned long long l = 0; l < p.n_layers; l++) {
        // Attention rmsnorm
        rmsnorm(s.xb, x, w.rms_att_weight + l * dim, dim);

        // Key and value point to the kv cache
        int loff = l * p.max_seq_len * kv_dim; // KV cache layer offset
        s.k = s.key_cache + loff + pos * kv_dim;
        s.v = s.value_cache + loff + pos * kv_dim;

        // Calculate QKV matrixes with matmuls between weight matrixes and inputs 
        matmul(s.q, s.xb, w.wq + l * dim * dim, dim, dim);
        matmul(s.k, s.xb, w.wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s.v, s.xb, w.wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE relative positional encoding
        RoPe_rotation(pos, s, dim, kv_dim, head_size);

        // Multihead attention
        multi_head_attention(pos, p, s, kv_dim, kv_mul, head_size, loff);

        // Final matmul to get the output of the attention
        matmul(s.xb2, s.xb, w.wo + l * dim * dim, dim, dim);

        // Residual connection back into x
        accum(x, s.xb2, dim);

        // FFN rmsnorm
        rmsnorm(s.xb, x, w.rms_ffn_weight + l * dim, dim);

        // FFN computation
        matmul(s.hb, s.xb, w.w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s.hb2, s.xb, w.w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        f_silu_elementwise_mul_w3(s, hidden_dim);

        // Final matmul to get the output of the FFN
        matmul(s.xb, s.hb, w.w2 + l * hidden_dim * dim, hidden_dim, dim);

        // Residual connection
        accum(x, s.xb, dim);
    }

    // Final rmsnorm
    rmsnorm(x, x, w.rms_final_weight, dim);

    // Classifier into logits
    matmul(s.logits_gpu, x, w.wcls, p.dim, p.vocab_size);
    CUDA_CHECK(cudaMemcpy(s.logits, s.logits_gpu, p.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    return s.logits;
}*/

}