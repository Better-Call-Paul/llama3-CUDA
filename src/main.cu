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

void generate_loop() {
    
}

}

int main(int argc, char *argv[]) {

    std::cout << "Stable" << "\n";
    // 32 layers 


    // input
    // embeddings
    // rms norm 
    // masked multi head attention
    // rms norm
    // feed forward + SwiGLU (activation function)
    // rms norm
    // linear 
    // softmax



    return 0;
}

