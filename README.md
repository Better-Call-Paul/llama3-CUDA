# llama3-pure-CUDA

## Overview
Llama 3 implemented in pure C++/CUDA.

The Llama 3 model is a transformer-based architecture designed for natural language processing tasks. This implementation aims to provide an efficient inference pipeline by utilizing custom CUDA kernels for better performance.

## Key Features:
- Transformer and Tokenizer components rewritten in C++ with CUDA.
- Custom CUDA kernels for all operations, avoiding dependencies on cuBLAS.
- Optimized matrix multiplication using vectorized loads (float4), tensor core operations, and block tiling.
- Softmax function optimized with block and warp reductions.
- Memory management and error checking for robustness.
- Command-line interface for easy use.

## Transformer Architecture
The transformer model consists of multiple layers, each containing self-attention and feed-forward networks (FFN). The implementation includes the following components:

- **Configuration (Config):** Stores hyperparameters such as model dimensions, number of layers, heads, etc.
- **Weights (TransformerWeights):** Holds the weights of the model, which are loaded into GPU memory.
- **Runtime State (RunState):** Maintains the state during inference, including activations and key-value caches.

### Key Components:
#### Self-Attention Mechanism:
- **Query, Key, Value Matrices:** Computed using custom matmul kernels.
- **RoPE (Rotary Position Embedding):** Applied to query and key vectors to incorporate positional information.
- **Multi-Head Attention:** Implemented with custom kernels to handle multiple attention heads efficiently.

#### Feed-Forward Network (FFN):
- **SwiGLU Activation:** Combines SiLU (Sigmoid Linear Unit) activation with gated linear units.
- **Custom Matmul Operations:** Optimized for tensor cores and vectorized operations.

#### Normalization Layers:
- **RMSNorm:** Custom kernel for Root Mean Square Layer Normalization, optimized using block reductions.

#### Output Projection:
- **Classifier Weights (wcls):** Used to project the final hidden state to the logits for token prediction.

## Tokenizer Implementation
The tokenizer translates between strings and token IDs using Byte Pair Encoding (BPE). It includes:

- **Vocabulary Loading:** Reads the vocabulary and scores from a binary file.
- **Token Encoding:** Converts input text to a sequence of token IDs, handling UTF-8 encoding and byte fallback.
- **Token Decoding:** Translates token IDs back to text, handling special tokens and raw bytes.

## Optimizations
### Matrix Multiplication
Techniques Used:
- **Vectorized Loads (float4):** Loads four float values at a time to improve memory throughput.
- **Tensor Core Operations:** Utilizes NVIDIA's tensor cores (where available) for mixed-precision matrix multiplications.
- **Block Tiling:** Divides the matrix into smaller tiles that fit into shared memory, reducing global memory accesses.
- **Thread Block Configuration:** Optimizes thread block sizes to maximize occupancy and performance.

Implementation Details:
- The `matmul_kernel` function is designed to perform matrix multiplication efficiently on the GPU.
- Shared memory is used to store tiles of the input matrices, enabling fast access and reducing the need for global memory reads.
- The kernel loops over the tiles of the matrices, accumulating the partial results.
- Supports large matrices by looping over multiple tiles if necessary.

### Softmax Function
Techniques Used:
- **Warp Reductions:** Further optimizes reductions by leveraging warp-level primitives where applicable.
- **Numerical Stability:** Subtracts the maximum value before exponentiation to prevent overflow issues.
- **Parallelization:** Distributes work across threads to compute exponentials and normalization efficiently.

Implementation Details:
- The `softmax_gpu` function is a device function called within CUDA kernels where softmax computation is needed.
- It handles the computation in a numerically stable manner, ensuring accurate results.
- The use of shared memory and efficient synchronization reduces the overhead and improves performance.

## Credits
Transformer and Tokenizer Design: The base class design for the transformer and tokenizer components is inspired by Andrej Karpathy's llama.c implementation. His work provided a foundation for structuring the model and handling tokenization efficiently.
