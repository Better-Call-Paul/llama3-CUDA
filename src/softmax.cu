#include "softmax.cuh"

namespace llama {

template<typename T>
__device__ void warpReduceMax(T& val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
}

template<typename T>
__device__ void warpReduceSum(T& val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
}

template<typename T>
__device__ T blockReduceMax(T val) {
    static __shared__ T shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceMax(val);

    if (lane == 0) shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : -Inf<T>();

    if (wid == 0) warpReduceMax(val);

    return val;
}

template<typename T>
__device__ T blockReduceSum(T val) {
    static __shared__ T shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    warpReduceSum(val);

    if (lane == 0) shared[wid] = val;

    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;

    if (wid == 0) warpReduceSum(val);

    return val;
}

template<typename T, int cols_per_thread>
__global__ void softmaxLocal(const T* input, T* output, size_t m, size_t n) {
    const int m_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    for (int64_t row = m_idx; row < m; row += gridDim.x * blockDim.y) {
        const int64_t row_offset = row * n;
        const T* row_x = input + row_offset;
        T* row_y = output + row_offset;
        T local_max = -Inf<T>();

        #pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = i * blockDim.x + tid;
            if (col < n) {
                local_max = max(local_max, row_x[col]);
            }
        }
        T s_max = blockReduceMax(local_max);

        T local_sum = 0;
        #pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = i * blockDim.x + tid;
            if (col < n) {
                T val = exp(row_x[col] - s_max);
                row_y[col] = val;
                local_sum += val;
            }
        }
        T s_sum = blockReduceSum(local_sum);

        #pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            const int col = i * blockDim.x + tid;
            if (col < n) {
                row_y[col] /= s_sum;
            }
        }
    }
}

template<typename T, int block_size>
__global__ void softmaxLarge(const T* input, T* output, size_t m, const size_t n) {
    const int m_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    for (int64_t row = m_idx; row < m; row += gridDim.x) {
        const int64_t row_offset = row * n;
        const T* row_x = input + row_offset;
        T* row_y = output + row_offset;

        T local_max = -Inf<T>();
        for (int col = tid; col < n; col += block_size) {
            local_max = max(local_max, row_x[col]);
        }
        T s_max = blockReduceMax(local_max);

        T local_sum = 0;
        for (int col = tid; col < n; col += block_size) {
            T val = exp(row_x[col] - s_max);
            row_y[col] = val;
            local_sum += val;
        }
        T s_sum = blockReduceSum(local_sum);

        for (int col = tid; col < n; col += block_size) {
            row_y[col] /= s_sum;
        }
    }
}

template<typename T>
__device__ T Inf() {
    if (std::is_same<T, float>::value) {
        return INFINITY;
    } else if (std::is_same<T, double>::value) {
        return INFINITY;
    } else if (std::is_same<T, half>::value) {
        return __float2half(INFINITY);
    }
    return T(0);  
}

// Explicit instantiations
template __global__ void softmaxLocal<float, 4>(const float*, float*, size_t, size_t);
template __global__ void softmaxLocal<double, 4>(const double*, double*, size_t, size_t);
template __global__ void softmaxLarge<float, 256>(const float*, float*, size_t, const size_t);
template __global__ void softmaxLarge<double, 256>(const double*, double*, size_t, const size_t);

// Explicit instantiations for block reduce functions
template __device__ float blockReduceMax<float>(float);
template __device__ double blockReduceMax<double>(double);
template __device__ float blockReduceSum<float>(float);
template __device__ double blockReduceSum<double>(double);

}  // namespace llama