#include "transformer.cuh"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include "rms_norm.cuh"
#include "rope_rotation.cuh"
#include "mat_mul.cuh"
#include "multi_head_attention.cuh"

namespace llama {

#define CUDA_CHECK(val) { \
    if (val != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(val)); \
    } \
}

RunState::RunState(const Config& config) {
    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    CUDA_CHECK(cudaMalloc(&x, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&xb, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&xb2, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hb, config.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&hb2, config.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&q, config.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&key_cache, config.n_layers * config.max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&value_cache, config.n_layers * config.max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&att, config.n_heads * config.max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&logits_gpu, config.vocab_size * sizeof(float)));
    logits = std::vector<float>(config.vocab_size, 0.0f);

    if (!x || !xb || !xb2 || !hb || !hb2 || !q || !key_cache || !value_cache || !att || !logits_gpu) {
        throw std::runtime_error("cudaMalloc failed!");
    }
}

RunState::~RunState() {
    cudaFree(x);
    cudaFree(xb);
    cudaFree(xb2);
    cudaFree(hb);
    cudaFree(hb2);
    cudaFree(q);
    cudaFree(att);
    cudaFree(logits_gpu);
    cudaFree(key_cache);
    cudaFree(value_cache);
}

Transformer::Transformer(const std::string& checkpoint_path) : fd(-1), data(nullptr), file_size(0) {
    read_checkpoint(checkpoint_path);
    state = std::make_unique<RunState>(config);
}

Transformer::~Transformer() noexcept {
    if (data != MAP_FAILED) { munmap(data, file_size); }
    if (fd != -1) { close(fd); }
    try {
        if (weights.token_embedding) {
            cudaFree(weights.token_embedding);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in destructor: " << e.what() << "\n";
    }
}

void Transformer::read_checkpoint(const std::string& checkpoint_path) {
    FILE* file = fopen(checkpoint_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Couldn't open file " + checkpoint_path);
    }

    if (fread(&config, sizeof(Config), 1, file) != 1) {
        throw std::runtime_error("Failed to read config");
    }

    bool shared_weights = config.vocab_size > 0;
    config.vocab_size = std::abs(config.vocab_size);

    fseek(file, 0, SEEK_END);
    file_size = ftell(file);
    fclose(file);

    fd = open(checkpoint_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("open failed!");
    }

    data = static_cast<float*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
        throw std::runtime_error("mmap failed!");
    }

    size_t weights_size = file_size - sizeof(Config);
    float* weights_ptr;
    CUDA_CHECK(cudaMalloc(&weights_ptr, weights_size));
    CUDA_CHECK(cudaMemcpy(weights_ptr, data + sizeof(Config) / sizeof(float), weights_size, cudaMemcpyHostToDevice));
    memory_map_weights(weights_ptr, shared_weights);
}

void Transformer::memory_map_weights(float* ptr, bool shared_weights) {
    int head_size = config.dim / config.n_heads;
    unsigned long long n_layers = config.n_layers;

    weights.token_embedding = ptr;
    ptr += config.vocab_size * config.dim;
    weights.rms_att_weight = ptr;
    ptr += n_layers * config.dim;
    weights.wq = ptr;
    ptr += n_layers * config.dim * (config.n_heads * head_size);
    weights.wk = ptr;
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
    weights.wv = ptr;
    ptr += n_layers * config.dim * (config.n_kv_heads * head_size);
    weights.wo = ptr;
    ptr += n_layers * (config.n_heads * head_size) * config.dim;
    weights.rms_ffn_weight = ptr;
    ptr += n_layers * config.dim;
    weights.w1 = ptr;
    ptr += n_layers * config.dim * config.hidden_dim;
    weights.w2 = ptr;
    ptr += n_layers * config.hidden_dim * config.dim;
    weights.w3 = ptr;
    ptr += n_layers * config.dim * config.hidden_dim;
    weights.rms_final_weight = ptr;
    ptr += config.dim;
    ptr += config.max_seq_len * head_size; // skip RoPE parameters
    weights.wcls = shared_weights ? weights.token_embedding : ptr;
}

std::vector<float> Transformer::forward(int token, int pos) {
    Config* p = &this->config;
    TransformerWeights* w = &this->weights;
    RunState* s = this->state.get();
    float* x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    // Copy the token embedding into x
    cudaMemcpy(x, w->token_embedding + token * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice);

    // Forward all the layers
    for (int l = 0; l < p->n_layers; l++) {
        // Attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);

        // QKV matmuls
        matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
        matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
        matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);

        // RoPE relative positional encoding
        rope_rotation(pos, s->q, s->k, dim, kv_dim, head_size);

        // Cache key and value
        int loff = l * p->max_seq_len * kv_dim;
        cudaMemcpy(s->key_cache + loff + pos * kv_dim, s->k, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(s->value_cache + loff + pos * kv_dim, s->v, kv_dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // Multihead attention
        multi_head_attention(pos, p->max_seq_len, s->q, s->key_cache + loff, s->value_cache + loff, 
                             s->att, s->xb, dim, kv_dim, p->n_heads, kv_mul, head_size);

        // Final matmul to get the output of the attention
        matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // FFN rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);

        // FFN
        matmul(s->hb, s->xb, w->w1 + l * dim * hidden_dim, dim, hidden_dim);
        matmul(s->hb2, s->xb, w->w3 + l * dim * hidden_dim, dim, hidden_dim);

        // SwiGLU non-linearity
        silu_elementwise_mul(s->hb, s->hb2, hidden_dim);

        // Final matmul to get the output of the ffn
        matmul(s->xb, s->hb, w->w2 + l * hidden_dim * dim, hidden_dim, dim);

        // Residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // Final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim);

    // Classifier into logits
    matmul(s->logits_gpu, x, w->wcls, p->dim, p->vocab_size);

    // Copy logits from GPU to CPU
    s->logits.resize(p->vocab_size);
    cudaMemcpy(s->logits.data(), s->logits_gpu, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);

    return s->logits;
}

}