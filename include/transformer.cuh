#ifndef TRANSFORMER_CUH
#define TRANSFORMER_CUH

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace llama {
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

} 
#endif 