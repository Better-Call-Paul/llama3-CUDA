// testing kernels

#include <iostream>

constexpr int max_stride = 8;

#define CHECK_CUDA_ERROR(call)
    {                                       \
        const cudaError_t error = call;     \
        if (error != cudaSuccess) {     
            fprintf("Error at {}, {}", );
            fprintf();
        }
    }                                       \


__global__ void cache_probing_kernel(int* data, int size, int stride) {

    // calculate indexes

    // iterate through memory

    // mess up compiler optimizer to measure true performance of hardware 
    extern __shared__ int shmem[];
    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x * threadIdx.y + blockIdx.x * blockIdx.y;

    

}


template<typename T>
struct gcd {

};



int main() {




    return 0;
}