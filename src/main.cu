#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

// Forward declarations of CUDA kernels
void rmsnorm(float *o, float *x, float *weight, int size);
void rope_rotation(int pos, float *q, float *k, int dim, int kv_dim, int head_size);
void multi_head_attention(int pos, int seq_len, float *q, float *k, float *v, float *att, float *out,
                          int dim, int kv_dim, int n_heads, int kv_mul, int head_size);
void silu_elementwise_mul(float *a, float *b, int size);

class Llama {
public:
    Llama(const std::string& model_path) {
        load_model(model_path);
        allocate_buffers();
        initialize_cublas();
    }

    ~Llama() {
        free_buffers();
        cublasDestroy(cublas_handle);
    }

    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens) {
        std::vector<int> output = prompt;
        
        for (int i = 0; i < max_new_tokens; ++i) {
            int next_token = forward(output);
            output.push_back(next_token);
            
            if (next_token == eos_token) break;
        }
        
        return output;
    }

private:
    // Model parameters
    int n_layers, dim, hidden_dim, n_heads, n_kv_heads, vocab_size, seq_len;
    int eos_token;

    // CUDA memory pointers
    float *token_embedding_cuda, *output_norm_cuda, *output_cuda;
    float *key_cache_cuda, *value_cache_cuda;
    std::vector<float*> layer_weights_cuda;

    // cuBLAS handle
    cublasHandle_t cublas_handle;

    void load_model(const std::string& model_path) {
        std::ifstream file(model_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open model file");
        }

        // Read model parameters
        file.read(reinterpret_cast<char*>(&n_layers), sizeof(int));
        file.read(reinterpret_cast<char*>(&dim), sizeof(int));
        file.read(reinterpret_cast<char*>(&hidden_dim), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_heads), sizeof(int));
        file.read(reinterpret_cast<char*>(&n_kv_heads), sizeof(int));
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&seq_len), sizeof(int));

        // Allocate and read model weights
        size_t embedding_size = vocab_size * dim * sizeof(float);
        std::vector<float> token_embedding(vocab_size * dim);
        file.read(reinterpret_cast<char*>(token_embedding.data()), embedding_size);
        cudaMalloc(&token_embedding_cuda, embedding_size);
        cudaMemcpy(token_embedding_cuda, token_embedding.data(), embedding_size, cudaMemcpyHostToDevice);

        // Allocate and load layer weights (simplified, you'll need to add all weight matrices)
        for (int i = 0; i < n_layers; ++i) {
            float* layer_weight;
            cudaMalloc(&layer_weight, dim * dim * sizeof(float));
            // Read and copy weight data
            layer_weights_cuda.push_back(layer_weight);
        }

        // Load output norm and other necessary weights
        // ...

        eos_token = 2;  // Assuming EOS token is 2, adjust if different
    }

    void allocate_buffers() {
        cudaMalloc(&output_norm_cuda, dim * sizeof(float));
        cudaMalloc(&output_cuda, dim * sizeof(float));
        cudaMalloc(&key_cache_cuda, n_layers * seq_len * dim * sizeof(float));
        cudaMalloc(&value_cache_cuda, n_layers * seq_len * dim * sizeof(float));
    }

    void free_buffers() {
        cudaFree(token_embedding_cuda);
        cudaFree(output_norm_cuda);
        cudaFree(output_cuda);
        cudaFree(key_cache_cuda);
        cudaFree(value_cache_cuda);
        for (auto& weight : layer_weights_cuda) {
            cudaFree(weight);
        }
    }

    void initialize_cublas() {
        cublasCreate(&cublas_handle);
    }

    int forward(const std::vector<int>& tokens) {
        int pos = tokens.size() - 1;
        int token = tokens.back();

        // Embed token
        float* x;
        cudaMalloc(&x, dim * sizeof(float));
        cudaMemcpy(x, token_embedding_cuda + token * dim, dim * sizeof(float), cudaMemcpyDeviceToDevice);

        // Forward through layers
        for (int l = 0; l < n_layers; ++l) {
            // Attention
            float* q, *k, *v, *att, *attn_out;
            cudaMalloc(&q, dim * sizeof(float));
            cudaMalloc(&k, dim * sizeof(float));
            cudaMalloc(&v, dim * sizeof(float));
            cudaMalloc(&att, n_heads * seq_len * sizeof(float));
            cudaMalloc(&attn_out, dim * sizeof(float));

            // Compute Q, K, V
            cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, dim, &alpha, layer_weights_cuda[l], dim, x, 1, &beta, q, 1);
            cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, dim, &alpha, layer_weights_cuda[l] + dim * dim, dim, x, 1, &beta, k, 1);
            cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, dim, &alpha, layer_weights_cuda[l] + 2 * dim * dim, dim, x, 1, &beta, v, 1);

            // Apply RoPE
            rope_rotation(pos, q, k, dim, dim, dim / n_heads);

            // Multi-head attention
            multi_head_attention(pos, seq_len, q, k, v, att, attn_out, dim, dim, n_heads, n_heads / n_kv_heads, dim / n_heads);

            // Add attention output to x
            float alpha = 1.0f, beta = 1.0f;
            cublasSaxpy(cublas_handle, dim, &alpha, attn_out, 1, x, 1);

            // Free temporary buffers
            cudaFree(q);
            cudaFree(k);
            cudaFree(v);
            cudaFree(att);
            cudaFree(attn_out);

            // Feed-forward network
            // (Simplified, you'll need to add the full FFN implementation)
            float* ffn_out;
            cudaMalloc(&ffn_out, dim * sizeof(float));
            silu_elementwise_mul(x, ffn_out, dim);
            cublasSaxpy(cublas_handle, dim, &alpha, ffn_out, 1, x, 1);
            cudaFree(ffn_out);
        }

        // Output norm
        rmsnorm(output_norm_cuda, x, output_cuda, dim);

        // Compute logits
        float* logits;
        cudaMalloc(&logits, vocab_size * sizeof(float));
        cublasSgemv(cublas_handle, CUBLAS_OP_T, dim, vocab_size, &alpha, token_embedding_cuda, dim, output_norm_cuda, 1, &beta, logits, 1);

        // Find max logit (argmax)
        int next_token;
        cublasIsamax(cublas_handle, vocab_size, logits, 1, &next_token);
        next_token -= 1;  // cublasIsamax returns 1-based index

        // Clean up
        cudaFree(x);
        cudaFree(logits);

        return next_token;
    }
};

// Main function
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <prompt>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];

    try {
        Llama model(model_path);

        // Tokenize prompt 
        std::vector<int> tokens;
        for (char c : prompt) {
            tokens.push_back(static_cast<int>(c));
        }

        // Generate
        std::vector<int> output = model.generate(tokens, 100);  // Generate up to 100 new tokens

        // Detokenize and print output 
        for (int token : output) {
            std::cout << static_cast<char>(token);
        }
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}