#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <cuda_runtime.h>
#include "bpe_tokenizer.cuh"
#include "transformer.cuh"
#include "utils.cuh"
#include "rms_norm.cuh"
#include "softmax.cuh"
#include "multi_head_attention.cuh"
#include "mat_mul.cuh"
#include <iostream>

namespace llama {

void generate(Transformer& transformer, Tokenizer& tokenizer, const std::string& prompt, int max_new_tokens) {
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
    
    if (prompt_tokens.empty()) {
        std::cerr << "Something is wrong, expected at least 1 prompt token\n";
        exit(EXIT_FAILURE);
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;

    while (pos < max_new_tokens - 1) {
        std::vector<float> logits = transformer.forward(token, pos);

        if (pos < prompt_tokens.size() - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample_argmax(logits.data(), transformer.config.vocab_size);
        }
        pos++;

        if (next == 1) break;

        std::string piece = tokenizer.decode(token, next);
        tokenizer.safe_printf(piece);
        std::cout.flush();
        
        token = next;

        if (start == 0) start = time_in_ms();
    }

    std::cout << std::endl;

    if (pos > 1) {
        long end = time_in_ms();
        std::cerr << "Token count: " << pos + 1 << ", elapsed: " << (float)(end - start) / 1000 
                  << "s, " << (int)((pos - 1) / (double)(end - start) * 1000) << " tokens/s\n";
    }
}
}
int main(int argc, char *argv[]) {
    std::string checkpoint_path = "../Meta-Llama-3-8B/consolidated.00.pth";
    std::string tokenizer_path = "../Meta-Llama-3-8B/tokenizer.bin";
    int max_new_tokens = 50;
    std::string prompt = "I have a dream";

    if (argc >= 2) prompt = argv[1];

    llama::Transformer transformer(checkpoint_path);
    if (max_new_tokens > transformer.config.max_seq_len)
        max_new_tokens = transformer.config.max_seq_len;

    llama::Tokenizer tokenizer(tokenizer_path, transformer.config.vocab_size);

    llama::generate(transformer, tokenizer, prompt, max_new_tokens);

    return 0;
}

