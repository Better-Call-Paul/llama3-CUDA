#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llama {

struct Config {
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int multiple_of;
    float ffn_dim_multiplier;
    float norm_eps;
    float rope_theta;
};

class TransformerWeights {

public:
    // token embedding table
    float *token_embedding_table; // (vocab_size, dim)
    // rmsnorm weights
    float *rms_att_weight; // (layer, dim) rmsnorm attention
    float *rms_ffnn_weight; // (layer, dim) feed forward norm

    // matmul matrix weights
    float *weight_q; // (layer, dim, n_heads * head_size)
    float *weight_k; // (layer, dim, kv_heads * head_size)
    float *weight_v; // (layer, dim, kv_heads * head_size)
    // output projection maatrix, weight matrix for multi head attention
    float *weight_o; // (layer, n_heads * head_size, dim)

    // feed forward neural network weights
    float *w1; // (layer, hidden_dim, dim);
    float *w2; // (layer, dim, hidden_dim);
    float *w3; // (layer, hidden_dim, dim) 

    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float *wcls;
};

class RunState {

public:

    RunState(const Config& config);
    ~RunState();

    // remove copy constructor
    RunState(const RunState&) = delete;
    // remove assignment operator
    RunState& operator=(const RunState&) = delete;

    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits_gpu; // output logits
    std::vector<int> logits; // host 
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
};



class Transformer {

public:
    Transformer(const std::string& checkpoint_path);
    ~Transformer();

    void memory_map_weights(TransformerWeights *weights, Config *config, float data_ptr*, int shared_weights);
    Config config;
    

private:

    TransformerWeights weights;
    RunState state;

    float *data; // memory mapped data ptr
    int fd; // file descriptor for memory mapping
    size_t file_size; // for checkpoint file in bytes
};



} 
#endif 