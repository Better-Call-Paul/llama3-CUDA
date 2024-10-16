#include "transformer.cuh"
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include "rms_norm.cuh"
#include "rope_rotation.cuh"
#include "mat_mul.cuh"
#include "multi_head_attention.cuh"

namespace llama {

#define CUDA_CHECK(val) { \
    if (val != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(val)); \
    } \
}


}