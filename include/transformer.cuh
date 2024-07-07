#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llama {

struct Config;
class TransformerWeights;

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