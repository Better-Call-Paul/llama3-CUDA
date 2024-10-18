// transformer inference

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

using namespace nvcuda::mma;


// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

const int num_threads_large = 1024;
const int num_threads_small = 64;

template<const uint BM, const uint BN, const uint BK>
__global__ void wmma_gemm(int M, int N, int K, float alpha, const __half* A, const __half *B, float beta, float *C) {

    // get block positions
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // global position
    const uint globalRow = threadIdx.y * WMMA_M;
    const uint globalCol = threadIdx.x * WMMA_N;

    // move matrices to current block
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // initialize output fragments
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    fill_fragment(c_frag, 0.0f);

    extern __shared__ __half shared_mem[];
    __half* A_shared = shared_mem;
    __half* B_shared = shared_mem + BM * BK;


    // loop with stride for gmem coalescing
    for (int tileIdx = 0; tileIdx < CEIL_DIV(K, WMMA_K); tileIdx++) {
        
       // Calculate the starting indices for A and B
        int aTileRow = threadIdx.y * WMMA_M;
        int aTileCol = tileIdx * WMMA_K + threadIdx.x * WMMA_K;

        int bTileRow = tileIdx * WMMA_K + threadIdx.y * WMMA_K;
        int bTileCol = threadIdx.x * WMMA_N;

        // Load A tile into shared memory
        for (int i = 0; i < WMMA_M; ++i) {
            int a_row = globalRow + i;
            int a_col = aTileCol;
            if (a_row < M && a_col < K) {
                A_shared[(threadIdx.y * WMMA_M + i) * BK + threadIdx.x * WMMA_K] =
                    A[a_row * lda + a_col];
            } else {
                A_shared[(threadIdx.y * WMMA_M + i) * BK + threadIdx.x * WMMA_K] = __float2half(0.0f);
            }
        }

        // Load B tile into shared memory
        for (int i = 0; i < WMMA_K; ++i) {
            int b_row = bTileRow + i;
            int b_col = globalCol;
            if (b_row < K && b_col < N) {
                B_shared[(threadIdx.y * WMMA_K + i) * BN + threadIdx.x * WMMA_N] =
                    B[b_row * ldb + b_col];
            } else {
                B_shared[(threadIdx.y * WMMA_K + i) * BN + threadIdx.x * WMMA_N] = __float2half(0.0f);
            }
        }

        __syncthreads();

        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> b_frag;

        // load inputs into fragments
        // frag, mem input, col size
        load_matrix_sync(a_frag, A_shared, BK);
        load_matrix_sync(b_frag, B_shared, BN);

        mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
        // dot prod
    }

    // Initialize C_frag with beta * C
    for (int i = 0; i < c_frag.num_elements; ++i) {
        int c_row = globalRow + (i / WMMA_N);
        int c_col = globalCol + (i % WMMA_N);
        if (c_row < M && c_col < N) {
            // Assuming row-major order for C
            c_frag.x[i] += beta * C_block[c_row * ldc + c_col];
        }
    }

    // Scale by alpha and store the result back to C_block
    for (int i = 0; i < WMMA_M; ++i) {
        for (int j = 0; j < WMMA_N; ++j) {
            int c_row = globalRow + i;
            int c_col = globalCol + j;
            if (c_row < M && c_col < N) {
                C_block[c_row * ldc + c_col] = alpha * c_frag.x[i * WMMA_N + j];
            }
        }
    }
    // load into c
}

// host gemm kernel caller

void matmul(half *x, half *w, float *output, int M, int N, int K) {
    // input is a vec so N is always 1
    const uint BK = 8;
    const uint BN = (M >= 128 && N >= 128) ? 128 : 64;
    const uint BM = (M >= 128 && N >= 128) ? 128 : 64;
    const uint TM = 8
    const uint TN = 8;

    dim3 gridDim = (CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim = ((BM * BN) / (TM * TN));
    wmma_gemm<BM, BN, BK>
    <<<gridDim, blockDim>>>(M, N, K, 1.0f, x, w, 1.0f, output);
}

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
__global__ void softmax(const uint K, const float* __restrict__ input) {
    
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
        max_val = fmaxf(max_val, input[i]);
    }

    // Find global max using block reduction
    max_val = block_reduce_max(max_val, shmem);
    
    float sum_exponents = 0.0f;
    // Compute partial sums of exponentials
    for (uint i = tid; i < K; i += total_threads) {
        float exponent = __expf(input[i] - max_val);
        sum_exponents += exponent;
        input[i] = exponent; 
    }

    // Find the sum of exponentials using block reduction
    sum_exponents = block_reduce_sum(sum_exponents, shmem);

    __syncthreads();
    
    // Normalize the exponentials to get softmax probabilities
    for (uint i = tid; i < K; i += total_threads) {
        input[i] /= sum_exponents;
    }
}

template<const uint size>
__global__ void rms_norm_kernel(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output, const uint elements_per_thread, const uint stride) {
    // square all vals in curr tile
    float ss = 0.0f;
    for (uint i = 0; i < elements_per_thread; ++i) {
        const uint curr_index = threadIdx.x + i * stride;
        if (curr_index < size) {
            ss += input[curr_index] * input[curr_index];
        }
    }

    extern __shared__ float shmem[];

    ss = block_reduce_sum(ss, shmem);

    __shared__ float global_ss;

    if (threadIdx.x == 0) {
        ss /= size;
        // avoid calling sqrt on negative value
        ss += 1e-5f;
        ss = 1.0f / sqrtf(ss);
        global_ss = ss;
    }

    __syncthreads();
    ss = global_ss;

    for (int i = 0; i < elements_per_thread; ++i) {
        int index = threadIdx.x + i * stride;
        if (index < size) {
            output[index] = weight[index] * (ss * input[index]);
        }
    }
}

void rms_norm(float* output, float *input, float* weight, uint size) {
    uint elements_per_thread = CEIL_DIV(size, num_threads_large);
    rms_norm_kernel<size>
    <<<1, num_threads_large>>>(input, weight, output, elements_per_thread, num_threads_large);
}





// ----------------------------------------------------------------------------
// Transformer model
// ----------------------------------------------------------------------------
typedef struct {
    int dim;            // D
    int hidden_dim;     // DD
    int n_layers;       // NL
    int n_heads;        // QHN, HN, HD = 48
    int n_kv_heads;     // KVHN = 6
    int vocab_size;     // VS
    int max_seq_len;    // M
} Config;

// CUDA NOTE: The TransformerWeights structure will be stored on the host,
// but all the pointers in the structure will point to data on the GPU.
// The checkpoint file is mmap-ed to the host and the weights portion
// is allocated on and copied to the GPU. Then, `memory_map_weights()` updates
// these structure pointers to point to the proper location. Happily, this
// function is the same for both C and CUDA.
typedef struct {
    float *token_embedding;     // (VS, D)
    float *rms_att_weight;      // (NL, D)
    float *wq;                  // (NL, D, D)
    float *wk;                  // (NL, D, D)
    float *wv;                  // (NL, D, D)
    float *wo;                  // (NL, D, D)
    float *rms_ffn_weight;      // (NL, D)
    float *w1;                  // (NL, DD, D)
    float *w2;                  // (NL, D, DD)
    float *w3;                  // (NL, DD, D)
    float *rms_final_weight;    // (D,)
    // (optional) classifier weights for the logits, on the last layer
    float *wcls;
} TransformerWeights;

// CUDA NOTE: The RunState structure will be stored on the host, but all the
// pointers in the structure will point to data on the GPU, created via
// cudaMalloc. The exception is logits which is the final result of the
// transformer & is copied from the GPU as the last step in the transformer
// and is used by the host.
typedef struct {
    // current wave of activations
    float *x;           // (D,) activation at current time stamp
    float *xb;          // (D,) same, but inside a residual branch
    float *xb2;         // (D,) an additional buffer just for convenience
    float *hb;          // (DD,) buffer for hidden dimension in the ffn
    float *hb2;         // (DD,) buffer for hidden dimension in the ffn
    float *q;           // (D,) query
    float *k;           // (D,) key
    float *v;           // (D,) value
    float *att;         // (HN, M) buffer for scores/attention values
    float *logits_gpu;  // output logits in GPU
    float *logits;      // output logits in CPU
    // kv cache
    float *key_cache;   // (NL, M, D)
    float *value_cache; // (NL, M, D)
} RunState;

typedef struct {
    Config config;              // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state;             // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd;                     // file descriptor for memory mapping
    float *data;                // memory mapped data pointer
    ssize_t file_size;          // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState *s, Config *p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    CUDA_CHECK(cudaMalloc((void **) &s->x, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->xb, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->xb2, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->hb, p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->hb2, p->hidden_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->q, p->dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->key_cache, p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->value_cache, p->n_layers * p->max_seq_len * kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->att, p->n_heads * p->max_seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **) &s->logits_gpu, p->vocab_size * sizeof(float)));
    // we calloc instead of malloc to keep valgrind happy
    s->logits = (float *) calloc(p->vocab_size, sizeof(float));

    // ensure all cudaMallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
        || !s->key_cache || !s->value_cache || !s->att || !s->logits_gpu || !s->logits) {
        fprintf(stderr, "cudaMalloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState *s) {
    CUDA_CHECK(cudaFree(s->x));
    CUDA_CHECK(cudaFree(s->xb));
    CUDA_CHECK(cudaFree(s->xb2));
    CUDA_CHECK(cudaFree(s->hb));
    CUDA_CHECK(cudaFree(s->hb2));
    CUDA_CHECK(cudaFree(s->q));
    CUDA_CHECK(cudaFree(s->att));
    CUDA_CHECK(cudaFree(s->logits_gpu));
    free(s->logits);
    CUDA_CHECK(cudaFree(s->key_cache));
    CUDA_CHECK(cudaFree(s->value_cache));
}

void memory_map_weights(TransformerWeights *w, Config *p, float *ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    w->token_embedding = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim;
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim;
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim;
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim;
    w->rms_final_weight = ptr;
    ptr += p->dim;
    ptr += p->max_seq_len * head_size / 2; // skip what used to be freq_cis_real (for RoPE)
    ptr += p->max_seq_len * head_size / 2; // skip what used to be freq_cis_imag (for RoPE)
    w->wcls = shared_weights ? w->token_embedding : ptr;
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights *weights,
                     int *fd, float **data, ssize_t *file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) {
        fprintf(stderr, "Couldn't open file %s\n", checkpoint);
        exit(EXIT_FAILURE);
    }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);
    // memory map the Transformer weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) {
        fprintf(stderr, "open failed!\n");
        exit(EXIT_FAILURE);
    }
    *data = (float *) mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) {
        fprintf(stderr, "mmap failed!\n");
        exit(EXIT_FAILURE);
    }
    // allocate & copy mmap data to the gpu first
    // to fit in the GPU, then copy the data only as needed while running.
    float *weights_ptr;
    size_t weights_size = *file_size - sizeof(Config);
    CUDA_CHECK(cudaMalloc((void **) &weights_ptr, weights_size));
    CUDA_CHECK(cudaMemcpy(weights_ptr, *data + sizeof(Config) / sizeof(float), weights_size, cudaMemcpyHostToDevice));
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}

void build_transformer(Transformer *t, char *checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // we cudaMalloc a region of memory, then hand the address to
    // the token_embedding field. Free it here.
    CUDA_CHECK(cudaFree(t->weights.token_embedding));
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
// ----------------------------------------------------------------------------
typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *) a)->str, ((TokenIndex *) b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size) {
    // i should have written the vocab_size into the tokenizer file... sigh
    t->vocab_size = vocab_size;
    // malloc space to hold the scores and the strings
    t->vocab = (char **) malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *) malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char) i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *) malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_scores);
    free(t->sorted_vocab);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char *) t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }

    // add additional processing to handle CJK characters
    int xff = 0xff;
    unsigned char fbit = (piece[0] & xff);
    unsigned char sbit = (piece[1] & xff);
    unsigned char mask = 0x40;

    switch (fbit) {
        case 0xC3:
            printf("%c", sbit | mask);
            break;
        case 0xC2:
            printf("%c", sbit);
            break;
        default:
            printf("%s", piece);
            break;
    }
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str}; // acts as the key to search for
    TokenIndex *res = (TokenIndex *) bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {
        fprintf(stderr, "cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    if (t->sorted_vocab == NULL) {
        // lazily malloc and sort the vocabulary
        t->sorted_vocab = (TokenIndex *) malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char *str_buffer = (char *) malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *) " ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char) str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (true) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}


int main(int argc, char* argv[]) {



    return 0;
}