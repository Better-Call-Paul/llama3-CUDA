// transformer_inference.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <cfloat>

// Use the WMMA namespace for convenience
using namespace nvcuda::wmma;

// Constants
#define FULL_MASK 0xFFFFFFFF
#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Error checking macro
#define CUDA_CHECK(val) { \
    cudaError_t err = (val); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error ") + std::to_string(err) + ": " + cudaGetErrorString(err) + " at " + __FILE__ + ":" + std::to_string(__LINE__)); \
    } \
}

// Namespace for custom kernels and functions
namespace llama {

// Warp reduce sum
__device__ float warp_reduce_sum(float sum) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    return sum;
}

// Warp reduce max
__device__ float warp_reduce_max(float max_val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_down_sync(FULL_MASK, max_val, offset));
    }
    return max_val;
}

// Block reduce sum
__device__ float block_reduce_sum(float sum, float* shmem) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE; // thread within warp index 
    
    // Reduce within warp
    sum = warp_reduce_sum(sum);
    
    // Write reduced sum to shared memory
    if (lane_id == 0) shmem[warp_id] = sum;
    
    __syncthreads();
    
    // Final reduce within first warp
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / WARP_SIZE) ? shmem[lane_id] : 0.0f);
        sum = warp_reduce_sum(sum);
    }
    
    __syncthreads();
    
    return sum;
}

// Block reduce max
__device__ float block_reduce_max(float max_val, float* shmem) {
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Reduce within warp
    max_val = warp_reduce_max(max_val);
    
    // Write reduced max to shared memory
    if (lane_id == 0) shmem[warp_id] = max_val;
    
    __syncthreads();
    
    // Final reduce within first warp
    if (warp_id == 0) {
        max_val = (lane_id < (blockDim.x / WARP_SIZE) ? shmem[lane_id] : -FLT_MAX);
        max_val = warp_reduce_max(max_val);
    }
    
    __syncthreads();
    
    return max_val;
}

/*
 * M x K Matrix Softmax Kernel
 */
__global__ void softmax_kernel(const uint M, const uint K, const __half* __restrict__ input, float* __restrict__ output) {
    
    extern __shared__ float shmem[];

    const uint row = blockIdx.x;
    // Each row computes one block

    // Move pointers to the current block's row
    input += row * K;
    output += row * K;

    const uint tid = threadIdx.x;
    const uint total_threads = blockDim.x; // stride for better GMEM coalescing 

    float max_val = -FLT_MAX;

    // Compute partial maxes
    for (uint i = tid; i < K; i += total_threads) {
        float val = __half2float(input[i]);
        max_val = fmaxf(max_val, val);
    }

    // Find global max using block reduction
    max_val = block_reduce_max(max_val, shmem);
    
    float sum_exponents = 0.0f;
    // Compute partial sums of exponentials
    for (uint i = tid; i < K; i += total_threads) {
        float curr_val = __half2float(input[i]);
        float exponent = __expf(curr_val - max_val);
        sum_exponents += exponent;
        output[i] = exponent; 
    }

    // Find the sum of exponentials using block reduction
    sum_exponents = block_reduce_sum(sum_exponents, shmem);

    __syncthreads();
    
    // Normalize the exponentials to get softmax probabilities
    for (uint i = tid; i < K; i += total_threads) {
        output[i] /= sum_exponents;
    }

}

/*
 * WMMA GEMM Kernel
 */
__global__ void wmma_gemm_kernel(int M, int N, int K, float alpha, const __half* A, const __half* B, float beta, float* C) {

    // Define the shape of the tile
    // Each block handles one tile of the output matrix

    // Identify the tile row and tile column
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    // Declare the fragments
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> C_frag;
    fill_fragment(C_frag, 0.0f);

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> A_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> B_frag;

    // Loop over tiles of K dimension
    for(int t = 0; t < K; t += WMMA_K) {
        // Load the inputs
        const __half* A_tile = A + tileRow * WMMA_M * K + t * WMMA_K;
        const __half* B_tile = B + t * WMMA_K * N + tileCol * WMMA_N;

        load_matrix_sync(A_frag, A_tile, K);
        load_matrix_sync(B_frag, B_tile, N);

        // Perform the matrix multiplication
        mma_sync(C_frag, A_frag, B_frag, C_frag);
    }

    // Load the C tile from memory
    float* C_tile = C + tileRow * WMMA_M * N + tileCol * WMMA_N;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> D_frag;
    load_matrix_sync(D_frag, C_tile, N);

    // Apply alpha and beta scaling
    for(int i = 0; i < C_frag.num_elements; i++) {
        C_frag.x[i] = alpha * C_frag.x[i] + beta * D_frag.x[i];
    }

    // Store the result back to C
    store_matrix_sync(C_tile, C_frag, N, mem_row_major);
}

/*
 * RMSNorm Kernel
 */
__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int size) {
    extern __shared__ float shmem[];

    // Compute the sum of squares
    float ss = 0.0f;
    for(int i = threadIdx.x; i < size; i += blockDim.x){
        ss += x[i] * x[i];
    }

    // Block reduction to compute total sum of squares
    ss = block_reduce_sum(ss, shmem);

    // Compute normalization factor
    float norm = 1.0f / sqrtf(ss / size + 1e-5f);

    // Normalize and scale
    for(int i = threadIdx.x; i < size; i += blockDim.x){
        o[i] = weight[i] * (x[i] * norm);
    }
}

/*
 * Element-wise addition kernel: C = A + B
 */
__global__ void add_elements_kernel(float* C, const float* A, const float* B, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        C[idx] = A[idx] + B[idx];
    }
}

/*
 * Weighted sum kernel: xb = sum(att * v) for each head
 */
__global__ void weighted_sum_kernel(float* xb, const float* att, const float* v, int n_heads, int max_seq_len, int kv_dim, int head_size) {
    int head = blockIdx.x;
    int idx = threadIdx.x;

    if(head >= n_heads || idx >= head_size){
        return;
    }

    float sum = 0.0f;
    for(int t = 0; t < max_seq_len; t++){
        float a = att[head * max_seq_len + t];
        float val = v[head * max_seq_len * kv_dim + t * kv_dim + idx];
        sum += a * val;
    }

    xb[head * head_size + idx] = sum;
}

/*
 * SiLU Activation and Element-Wise Multiplication
 */
__global__ void silu_elementwise_mul_kernel(float* hb, const float* hb2, int hidden_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < hidden_dim){
        float silu = hb[idx] * (1.0f / (1.0f + expf(-hb[idx])));
        hb[idx] = silu * hb2[idx];
    }
}

/*
 * Float to Half Conversion Kernel
 */
__global__ void float_to_half_kernel(const float* __restrict__ input, __half* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = __float2half(input[idx]);
    }
}

/*
 * Half to Float Conversion Kernel
 */
__global__ void half_to_float_kernel(const __half* __restrict__ input, float* __restrict__ output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        output[idx] = __half2float(input[idx]);
    }
}

// Host functions to launch kernels

void softmax(const uint M, const uint K, const __half* input, float* output) {
    // Assuming block size is 256 threads
    int threads = 256;
    int blocks = M;
    size_t shared_mem = threads * sizeof(float);
    softmax_kernel<<<blocks, threads, shared_mem>>>(M, K, input, output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void wmma_gemm(int M, int N, int K, float alpha, const __half* A, const __half* B, float beta, float* C) {
    dim3 grid((N + WMMA_N - 1)/WMMA_N, (M + WMMA_M - 1)/WMMA_M);
    dim3 block(1,1);
    wmma_gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void rmsnorm_custom(float* o, const float* x, const float* weight, int size) {
    int threads = 256;
    int blocks = 1;
    size_t shared_mem = (threads / WARP_SIZE) * sizeof(float);
    rmsnorm_kernel<<<blocks, threads, shared_mem>>>(o, x, weight, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void add_elements(float* C, const float* A, const float* B, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_elements_kernel<<<blocks, threads>>>(C, A, B, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void weighted_sum(float* xb, const float* att, const float* v, int n_heads, int max_seq_len, int kv_dim, int head_size) {
    dim3 grid(n_heads);
    dim3 block(head_size);
    weighted_sum_kernel<<<grid, block>>>(xb, att, v, n_heads, max_seq_len, kv_dim, head_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void silu_elementwise_mul(float* hb, const float* hb2, int hidden_dim) {
    int threads = 256;
    int blocks = (hidden_dim + threads -1)/threads;
    silu_elementwise_mul_kernel<<<blocks, threads>>>(hb, hb2, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void convert_float_to_half(const float* d_input, __half* d_output, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    float_to_half_kernel<<<blocks, threads>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void convert_half_to_float(const __half* d_input, float* d_output, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    half_to_float_kernel<<<blocks, threads>>>(d_input, d_output, size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace llama

// Data structures

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
};

struct TransformerWeights {
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

struct RunState {
    float* x;
    float* xb;
    float* xb2;
    float* hb;
    float* hb2;
    float* q;
    float* k;
    float* v;
    float* att;
    float* logits_gpu;
    float* logits; // Host-side
    float* key_cache;
    float* value_cache;
};

// Tokenizer class
class Tokenizer {
public:
    Tokenizer(const std::string& tokenizer_path, int vocab_size);
    ~Tokenizer();

    std::vector<int> encode(const std::string& text, bool bos = true, bool eos = true);
    std::string decode(int prev_token, int token) const;

private:
    void build_tokenizer(const std::string& tokenizer_path, int vocab_size);
    int str_lookup(const std::string& str) const;
    void sort_vocab();

    std::vector<std::string> vocab_;
    std::vector<float> vocab_scores_;
    std::vector<std::pair<std::string, int>> sorted_vocab_;
    int vocab_size_;
    unsigned int max_token_length_;
    unsigned char byte_pieces_[512];
};

Tokenizer::Tokenizer(const std::string& tokenizer_path, int vocab_size)
    : vocab_size_(vocab_size), max_token_length_(0) {
    // Initialize byte_pieces
    for(int i =0; i<256; i++) {
        byte_pieces_[i *2] = static_cast<unsigned char>(i);
        byte_pieces_[i *2 +1] = '\0';
    }

    build_tokenizer(tokenizer_path, vocab_size);
}

Tokenizer::~Tokenizer() {
    // Nothing to free
}

void Tokenizer::build_tokenizer(const std::string& tokenizer_path, int vocab_size) {
    std::ifstream file(tokenizer_path, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Couldn't load tokenizer file: " + tokenizer_path);
    }

    // Read max_token_length
    file.read(reinterpret_cast<char*>(&max_token_length_), sizeof(int));
    if(file.fail()) {
        throw std::runtime_error("Failed to read max_token_length from tokenizer file");
    }

    vocab_.resize(vocab_size_);
    vocab_scores_.resize(vocab_size_);

    for(int i = 0; i < vocab_size_; i++) {
        // Read score
        file.read(reinterpret_cast<char*>(&vocab_scores_[i]), sizeof(float));
        if(file.fail()) {
            throw std::runtime_error("Failed to read vocab score");
        }

        // Read token length
        int len;
        file.read(reinterpret_cast<char*>(&len), sizeof(int));
        if(file.fail()) {
            throw std::runtime_error("Failed to read token length");
        }

        // Read token string
        std::string token(len, '\0');
        file.read(&token[0], len);
        if(file.fail()) {
            throw std::runtime_error("Failed to read token string");
        }

        // Store with null terminator
        token.push_back('\0');
        vocab_[i] = token;
    }

    file.close();

    sort_vocab();
}

void Tokenizer::sort_vocab() {
    sorted_vocab_.reserve(vocab_size_);
    for(int i =0; i < vocab_size_; i++) {
        sorted_vocab_.emplace_back(std::make_pair(vocab_[i], i));
    }

    std::sort(sorted_vocab_.begin(), sorted_vocab_.end(), [&](const std::pair<std::string, int>& a, const std::pair<std::string, int>& b) -> bool {
        return a.first < b.first;
    });
}

int Tokenizer::str_lookup(const std::string& str) const {
    // Binary search in sorted_vocab_
    int left = 0;
    int right = sorted_vocab_.size() -1;
    while(left <= right) {
        int mid = left + (right - left)/2;
        if(sorted_vocab_[mid].first == str) {
            return sorted_vocab_[mid].second;
        }
        else if(sorted_vocab_[mid].first < str) {
            left = mid +1;
        }
        else {
            right = mid -1;
        }
    }
    return -1;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) {
    std::vector<int> tokens;
    if(bos) {
        tokens.push_back(1); // BOS token
    }

    // Simplified encoding: split into words and lookup
    // Implement proper BPE encoding as needed

    std::string current;
    for(auto c: text) {
        current += c;
        int id = str_lookup(current);
        if(id != -1) {
            tokens.push_back(id);
            current.clear();
        }
    }

    // Handle remaining
    if(!current.empty()) {
        // Byte fallback: encode each byte as token (start at index 3)
        for(auto c: current) {
            tokens.push_back(static_cast<unsigned char>(c) + 3);
        }
    }

    if(eos) {
        tokens.push_back(2); // EOS token
    }

    return tokens;
}

std::string Tokenizer::decode(int prev_token, int token) const {
    if(token <0 || token >= vocab_size_){
        return "";
    }

    std::string piece = vocab_[token];
    // Following BOS (1) token, sentencepiece decoder strips any leading whitespace
    if(prev_token ==1 && piece[0] == ' ') {
        piece = piece.substr(1);
    }

    // Handle byte_fallback encoding
    if(piece.size() ==0){
        return "";
    }
    if(piece.size() ==1){
        unsigned char byte_val = piece[0];
        if(!(isprint(byte_val) || isspace(byte_val))){
            // Return empty string for non-printable
            return "";
        }
    }

    // Check if token is in format <0xXX>
    if(piece.size() ==7 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[6] == '>'){
        unsigned char byte_val;
        sscanf(piece.c_str(), "<0x%02hhX>", &byte_val);
        return std::string(reinterpret_cast<const char*>(byte_pieces_) + byte_val *2);
    }

    return piece;
}

// Transformer class
class Transformer {
public:
    Transformer(const std::string& checkpoint_path);
    ~Transformer();

    float* forward(int token, int pos);

private:
    void build_transformer(const std::string& checkpoint_path);
    void malloc_run_state();
    void free_run_state();
    void memory_map_weights(__half* ptr, int shared_weights);
    void read_checkpoint(const std::string& checkpoint_path);

    Config config_;
    TransformerWeights weights_;
    RunState state_;
    int fd_;
    float* data_;
    ssize_t file_size_;
};

Transformer::Transformer(const std::string& checkpoint_path)
    : fd_(-1), data_(nullptr), file_size_(0) {
    build_transformer(checkpoint_path);
    malloc_run_state();
}

Transformer::~Transformer() {
    // Free RunState buffers
    free_run_state();

    // Unmap and close the checkpoint file
    if(data_ != MAP_FAILED && data_ != nullptr){
        munmap(data_, file_size_);
    }
    if(fd_ != -1){
        close(fd_);
    }

    // Free weights (assume allocated as __half*, but stored as float*)
    cudaFree(weights_.token_embedding);
    cudaFree(weights_.rms_att_weight);
    cudaFree(weights_.wq);
    cudaFree(weights_.wk);
    cudaFree(weights_.wv);
    cudaFree(weights_.wo);
    cudaFree(weights_.rms_ffn_weight);
    cudaFree(weights_.w1);
    cudaFree(weights_.w2);
    cudaFree(weights_.w3);
    cudaFree(weights_.rms_final_weight);
    cudaFree(weights_.wcls);
}

void Transformer::read_checkpoint(const std::string& checkpoint_path) {
    std::ifstream file(checkpoint_path, std::ios::binary);
    if(!file.is_open()) {
        throw std::runtime_error("Couldn't open checkpoint file: " + checkpoint_path);
    }

    // Read Config
    file.read(reinterpret_cast<char*>(&config_), sizeof(Config));
    if(file.fail()){
        throw std::runtime_error("Failed to read Config from checkpoint");
    }

    // Determine shared_weights based on vocab_size
    int shared_weights = config_.vocab_size >0 ?1 :0;
    config_.vocab_size = abs(config_.vocab_size);

    // Get file size
    file.seekg(0, std::ios::end);
    file_size_ = file.tellg();
    file.seekg(0, std::ios::beg);
    file.close();

    // Memory map the checkpoint file
    fd_ = open(checkpoint_path.c_str(), O_RDONLY);
    if(fd_ == -1){
        throw std::runtime_error("Failed to open checkpoint file for memory mapping");
    }

    data_ = reinterpret_cast<float*>(mmap(NULL, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if(data_ == MAP_FAILED){
        close(fd_);
        throw std::runtime_error("Failed to memory map checkpoint file");
    }

    // Allocate GPU memory for weights and copy data
    size_t weights_size = file_size_ - sizeof(Config);
    __half* weights_ptr_half;
    CUDA_CHECK(cudaMalloc(&weights_ptr_half, weights_size));
    // Assuming weights in checkpoint are in float
    float* weights_host = new float[weights_size / sizeof(float)];
    memcpy(weights_host, data_ + (sizeof(Config) / sizeof(float)), weights_size);
    
    // Convert float to half
    __half* weights_host_half = new __half[weights_size / sizeof(__half)];
    for(size_t i =0; i < weights_size / sizeof(__half); i++) {
        weights_host_half[i] = __float2half(weights_host[i]);
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(weights_ptr_half, weights_host_half, weights_size, cudaMemcpyHostToDevice));

    delete[] weights_host;
    delete[] weights_host_half;

    memory_map_weights(weights_ptr_half, shared_weights);
}

void Transformer::memory_map_weights(__half* ptr, int shared_weights) {
    // Map the weights according to the order in TransformerWeights
    // Adjust this based on actual weight layout in checkpoint

    // token_embedding: (vocab_size, dim)
    weights_.token_embedding = reinterpret_cast<float*>(ptr);
    ptr += config_.vocab_size * config_.dim;

    // rms_att_weight: (n_layers, dim)
    weights_.rms_att_weight = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.dim;

    // wq: (n_layers, dim, dim)
    weights_.wq = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.dim * config_.dim;

    // wk: (n_layers, dim, dim)
    weights_.wk = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.dim * config_.dim;

    // wv: (n_layers, dim, dim)
    weights_.wv = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.dim * config_.dim;

    // wo: (n_layers, dim, dim)
    weights_.wo = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.dim * config_.dim;

    // rms_ffn_weight: (n_layers, dim)
    weights_.rms_ffn_weight = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.dim;

    // w1: (n_layers, dim, hidden_dim)
    weights_.w1 = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.dim * config_.hidden_dim;

    // w2: (n_layers, hidden_dim, dim)
    weights_.w2 = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.hidden_dim * config_.dim;

    // w3: (n_layers, hidden_dim, dim)
    weights_.w3 = reinterpret_cast<float*>(ptr);
    ptr += config_.n_layers * config_.hidden_dim * config_.dim;

    // rms_final_weight: (dim)
    weights_.rms_final_weight = reinterpret_cast<float*>(ptr);
    ptr += config_.dim;

    // wcls: (dim, vocab_size)
    weights_.wcls = reinterpret_cast<float*>(ptr);
}

void Transformer::build_transformer(const std::string& checkpoint_path) {
    read_checkpoint(checkpoint_path);
    // Additional initialization if necessary
}

void Transformer::malloc_run_state() {
    // Allocate GPU memory for RunState
    CUDA_CHECK(cudaMalloc(&state_.x, config_.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.xb, config_.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.xb2, config_.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.hb, config_.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.hb2, config_.hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.q, config_.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.k, config_.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.v, config_.dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.att, config_.n_heads * config_.max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.logits_gpu, config_.vocab_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.key_cache, config_.n_layers * config_.max_seq_len * ((config_.dim * config_.n_kv_heads) / config_.n_heads) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&state_.value_cache, config_.n_layers * config_.max_seq_len * ((config_.dim * config_.n_kv_heads) / config_.n_heads) * sizeof(float)));

    // Allocate host-side logits
    state_.logits = new float[config_.vocab_size];
    memset(state_.logits, 0, config_.vocab_size * sizeof(float));
}

void Transformer::free_run_state() {
    // Free GPU memory
    cudaFree(state_.x);
    cudaFree(state_.xb);
    cudaFree(state_.xb2);
    cudaFree(state_.hb);
    cudaFree(state_.hb2);
    cudaFree(state_.q);
    cudaFree(state_.k);
    cudaFree(state_.v);
    cudaFree(state_.att);
    cudaFree(state_.logits_gpu);
    cudaFree(state_.key_cache);
    cudaFree(state_.value_cache);

    // Free host-side logits
    delete[] state_.logits;
}

/*
 * Helper function to sample argmax
 */
int sample_argmax(float* probabilities, int n){
    int max_i = 0;
    float max_p = probabilities[0];
    for(int i =1; i <n; i++){
        if(probabilities[i] > max_p){
            max_p = probabilities[i];
            max_i = i;
        }
    }
    return max_i;
}

// Generator class to handle text generation
struct Generator {
    Transformer& transformer;
    Tokenizer& tokenizer;

    Generator(Transformer& t, Tokenizer& tok) : transformer(t), tokenizer(tok) {}

    void generate(const std::string& prompt, int max_new_tokens){
        // Encode prompt
        std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
        if(prompt_tokens.empty()){
            throw std::runtime_error("Prompt encoding resulted in zero tokens");
        }

        // Start generation
        int pos =0;
        int token = prompt_tokens[0];
        while(pos < max_new_tokens){
            // Forward pass
            float* logits = transformer.forward(token, pos);

            // Decide next token
            int next;
            if(pos < prompt_tokens.size() -1){
                next = prompt_tokens[pos +1];
            }
            else{
                next = sample_argmax(logits, transformer.config_.vocab_size);
            }

            pos++;

            // Termination condition
            if(next ==1){ // BOS token
                break;
            }

            // Decode and print
            std::string piece = tokenizer.decode(token, next);
            std::cout << piece;
            std::cout.flush();

            token = next;
        }
        std::cout << std::endl;

        // Report tokens per second
        // For simplicity, timing is omitted here
    }
};

// Main function
int main(int argc, char* argv[]){
    try{
        // Default parameters
        std::string checkpoint_path = "stories15M.bin";
        std::string tokenizer_path = "tokenizer.bin";
        int max_new_tokens = 50;
        std::string prompt = "I have a dream";

        // Parse command line arguments
        if(argc >=2){
            prompt = argv[1];
        }
        if(argc >=3){
            max_new_tokens = std::stoi(argv[2]);
        }

        // Initialize Transformer
        Transformer transformer(checkpoint_path);
        if(max_new_tokens > transformer.config_.max_seq_len){
            max_new_tokens = transformer.config_.max_seq_len;
        }

        // Initialize Tokenizer
        Tokenizer tokenizer(tokenizer_path, transformer.config_.vocab_size);

        // Initialize Generator
        Generator generator(transformer, tokenizer);

        // Generate text
        generator.generate(prompt, max_new_tokens);
    }
    catch(const std::exception& e){
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
