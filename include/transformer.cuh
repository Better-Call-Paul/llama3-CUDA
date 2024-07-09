#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llama {

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

class RunState {
public:
    RunState(const Config& config);
    ~RunState();
    
    RunState(const RunState&) = delete;
    RunState& operator=(const RunState&) = delete;
  
private:
    float *x, *xb, *xb2, *hb, *hb2, *q, *k, *v, *att, *logits_gpu;
    std::vector<float> logits;
    float *key_cache, *value_cache;
};

class Transformer {
public:
    Transformer(const std::string& checkpoint_path);
    ~Transformer();

    Transformer(const Transformer&) = delete;
    Transformer& operator=(const Transformer&) = delete;

private:
    void read_checkpoint(const std::string& checkpoint_path);
    void memory_map_weights(float* ptr, bool shared_weights);

    Config config;
    TransformerWeights weights;
    std::unique_ptr<RunState> state;
    int fd;
    float* data;
    size_t file_size;
};

} 
#endif 