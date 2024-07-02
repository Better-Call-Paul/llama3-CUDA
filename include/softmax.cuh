#ifndef MAT_OPERATIONS_CUH
#define MAT_OPERATIONS_CUH


namespace llama {

template<typename T, int cols_per_thread>
__inline__ __device__ void llama::warpReduceMax(T *val, int warp_size);

template<typename T, int cols_per_thread>
__inline__ __device__ void llama::warpReduceSum(T* val, int warp_size);

template<typename T, int cols_per_thread>
__inline__ __device__ void llama::blockReduceSum(T* val, int warp_size);

template<int cols_per_thread>
__global void softmaxLocal(const T* input, T* output, size_t m, size_t n);

template<int block_size>
__global void softmaxLarge(const T* input, T* output, size_t m, const size_t n);

}

#endif