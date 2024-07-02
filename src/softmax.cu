#include "softmax.cuh"

namespace llama {

template<typename T, int cols_per_thread>
__inline__ __device__ void warpReduceMax(T *val, int warp_size) {
    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        #pragma unroll
        for (int j = warp_size / 2; j > 0; j >>= 1) {
            val[i] = max(val[i], __shfl_xor_sync(FULL_MASK, val[i], j, warp_size));
        }
    }
}

template<typename T, int cols_per_thread>
__inline__ __device__ void warpReduceSum(T* val, int warp_size) {
    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        #pragma unroll
        for (int j = warp_size / 2; j > 0; j >>= 1) {
            val[i] += __shfl_xor_sync(FULL_MASK, val[i], j, warp_size);
        }
    }
}

template<typename T, int cols_per_thread>
__inline__ __device__ void blockReduceSum(T* val, int warp_size) {
    int lane = threadIdx.x % warp_size;
    int wid = threadIdx.x / warp_size;
    __shared__ T shared[warp_size + 1][cols_per_thread];

    warpReduceSum<T, cols_per_thread>(val, warp_size);
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            shared[wid][i] = val[i];
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        val[i] = (threadIdx.x < (blockDim.x / warp_size)) ? shared[lane][i] : static_cast<T>(0);
    }

    if (wid == 0) {
        warpReduceSum<T, cols_per_thread>(val, warp_size);
    }
}

template<typename T, int cols_per_thread>
__global__ void softmaxLocal(const T* input, T* output, size_t m, size_t n) {
    constexpr int num_elements = cols_per_thread;
    T buf[num_elements];
    const int m_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    for (int64_t row = m_idx; row < m; row += gridDim.x * blockDim.y) {
        const int64_t row_offset = row * n;
        const T* row_x = input + row_offset;
        T* row_y = output + row_offset;
        T local_max = -Inf<T>();

        #pragma unroll
        for (int i = 0; i < num_elements; ++i) {
            const int col = i * blockDim.x + tid;
            if (col < n) {
                buf[i] = row_x[col];
                local_max = max(local_max, buf[i]);
            } else {
                buf[i] = -Inf<T>();
            }
        }
        warpReduceMax<T, num_elements>(&local_max, blockDim.x);

        T local_sum = 0;
        #pragma unroll
        for (int i = 0; i < num_elements; ++i) {
            buf[i] = exp(buf[i] - local_max);
            local_sum += buf[i];
        }
        warpReduceSum<T, 1>(&local_sum, blockDim.x);

        #pragma unroll
        for (int i = 0; i < num_elements; ++i) {
            const int col = i * blockDim.x + tid;
            if (col < n) {
                row_y[col] = buf[i] / local_sum;
            }
        }
    }
}

template<typename T, int block_size>
__global__ void softmaxLarge(const T* input, T* output, size_t m, const size_t n) {
    const int m_idx = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ T shared_buf[];
    T* buf = shared_buf;

    for (int64_t row = m_idx; row < m; row += gridDim.x) {
        const int64_t row_offset = row * n;
        const T* row_x = input + row_offset;
        T* row_y = output + row_offset;
        T local_max = -Inf<T>();

        for (int col = tid; col < n; col += blockDim.x) {
            buf[col] = row_x[col];
            local_max = max(local_max, buf[col]);
        }
        blockReduceMax<T, 1>(&local_max, blockDim.x);

        __shared__ T s_max;
        if (threadIdx.x == 0) {
            s_max = local_max;
        }
        __syncthreads();

        T local_sum = 0;
        for (int i = tid; i < n; i += blockDim.x) {
            T local_val = exp(buf[i] - s_max);
            buf[i] = local_val;
            local_sum += local_val;
        }
        blockReduceSum<T, 1>(&local_sum, blockDim.x);

        __shared__ T s_sum;
        if (threadIdx.x == 0) {
            s_sum = local_sum;
        }
        __syncthreads();

        for (int i = tid; i < n; i += blockDim.x) {
            row_y[i] = buf[i] / s_sum;
        }
    }
}

// Helper function to get the appropriate type for infinity
template<typename T>
__device__ T Inf() {
    if (std::is_same<T, float>::value) {
        return INFINITY;
    } else if (std::is_same<T, double>::value) {
        return INFINITY;
    } else if (std::is_same<T, half>::value) {
        return __float2half(INFINITY);
    }
    // Add more types if needed
    return T(0);  
}

}