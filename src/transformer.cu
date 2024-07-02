#include "transformer.cuh"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace llama {

#define CUDA_CHECK(val) { \
    if (val != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(val)); \
    } \
}

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
};

class TransformerWeights {
public:
    float* token_embedding;
    float* rms_att_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
};

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

Transformer::~Transformer() {
    if (data != MAP_FAILED) { munmap(data, file_size); }
    if (fd != -1) { close(fd); }
    CUDA_CHECK(cudaFree(weights.token_embedding));
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

}