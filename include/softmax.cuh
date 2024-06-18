#ifndef SOFTMAX_CUH
#define SOFTMAX_CUH

#define BLOCK_SIZE 1024

template<typename T, int N>
__global__ half4 warp_reduce();

template<typename T, int N>
__global__ void softmax_small(const half4* input, half4* output, size_t n, size_t m);



template<typename T, int N>
__global__ void softmax_large(const half4* input, half4* output, size_t n, size_t m);


#endif