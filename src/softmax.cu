#include "softmax.cuh"

namespace llama {

template<typename T, int cols_per_thread>
__inline__ __device__ void llama::warpReduceMax(T *val, int warp_size) {
    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        #pragma unroll
        for (int j = warp_size / 2; j > 0; j >>= 1) {
            val[i] = max(val[i], __shfl_xor_sync(FULL_MASK, val[i], j, warp_size));
        }
    }
}


template<typename T, int cols_per_thread>
__inline__ __device__ void llama::warpReduceSum(T* val, int warp_size) {
    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        #pragma unroll
        for (int j = warp_size / 2; j > 0; j >>= 1) {
            val[i] += __shfl_xor_sync(FULL_MASK, val[i], j, warp_size);
        }
    }
}

template<typename T, int cols_per_thread>
__inline__ __device__ void llama::blockReduceSum(T* val, int warp_size) {
    int lane = threadIdx.x % warp_size; // lane id within warp 
    int wid = threadIdx.x / warp_size; // warp id
    __shared__ T shared[warp_size + 1];

    warpReduceSum<T, cols_per_thread>(val, warp_size);
    if (lane == 0) { // final value
        #pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            shared[i][wid] = val[i];
        }
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        val[i] = (threadIdx.x < (blockDim.x  / 32.f)) ? shared[i][lane] : static_cast<T>(0.0f); 
    }

    if (wid == 0) {
        warpReduceSum<T, numELements>(val, warp_size);
    }

}

template<int cols_per_thread>
      __global void softmaxLocal(const half4* input, half4* output, size_t m, size_t n) {
          constexpr int num_packs = (cols_per_thread+3) / 4;//pack_size = 4, k/32 = cols_per_thread, num_packs = k/32/4
          float4 buf[num_packs];
          const int m_idx = blockIdx.x * blockDim.y + threadIdx.y;//blockDim.y=4=thread_group_per_block
          const int tid = threadIdx.x;

          for (int64_t row = m_idx; row < m; row += gridDim.x * blockDim.y) {

              const int64_t row_offset = row * (n >> 2);
              const half4* row_x = input + row_offset;
              half4* row_y = output + row_offset;
              float local_max[1] = {-Inf<float>()};
              #pragma unroll
              for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
                  const int col = pack_id * blockDim.x + tid;
                  // row_y[col] = row_x[col];
                  if (col < n/4) {
                      buf[pack_id] = {
                          half2float(row_x[col].x),
                          __half2float(row_x[col].y),
                          __half2float(row_x[col].z),
                          __half2float(row_x[col].w)};
                      local_max[0] = max(local_max[0], max(max(buf[pack_id].x, buf[pack_id].y), max(buf[pack_id].z, buf[pack_id].w)));
                  } else {
                      buf[pack_id].x = -Inf<float>();
                      buf[pack_id].y = -Inf<float>();
                      buf[pack_id].z = -Inf<float>();
                      buf[pack_id].w = -Inf<float>();
                  }
              }
              warpReduceMax<float,1>(local_max, blockDim.x);

              float local_sum[1] = {0.0f};
              #pragma unroll
              for (int i = 0; i < num_packs; ++i) {
                  buf[i].x = exp(buf[i].x - local_max[0]);
                  buf[i].y = exp(buf[i].y - local_max[0]);
                  buf[i].z = exp(buf[i].z - local_max[0]);
                  buf[i].w = exp(buf[i].w - local_max[0]);
                  local_sum[0] += buf[i].x;
                  local_sum[0] += buf[i].y;
                  local_sum[0] += buf[i].z;
                  local_sum[0] += buf[i].w;
              }
              warpReduceSum<float, 1>(local_sum, blockDim.x);

              for (int i = 0; i < num_packs; ++i) {
                  const int col = i * blockDim.x + tid;
                  if (col < n/4) {
                      row_y[col] = { buf[i].x/local_sum[0], buf[i].y/local_sum[0], buf[i].z/local_sum[0], buf[i].w/local_sum[0] };
                  }
              }
          }
      } 

      template<int block_size>
      __global void softmaxLarge(
          const half4* input,
          half4* output,
          size_t m,
          const size_t n) {
          const int m_idx = blockIdx.x;
          const int tid = threadIdx.x;
          extern shared align(sizeof(float)) unsigned char shared_buf[];//size_t smem = nsizeof(float)
          auto buf = reinterpret_cast<float*>(shared_buf);
          const int num_packs = n >> 2;
          for (int64_t row = m_idx; row < m; row += gridDim.x) {
              const int64_t row_offset = row  (n>>2);
              const half4* row_x = input + row_offset;
              half4* row_y = output + row_offset;
              float local_max[1] = {-Inf<float>()};

              for (int pack_id = tid; pack_id < num_packs; pack_id += blockDim.x) {
                  const int col = pack_id;
                  // store to local register, which is faster than shared memory
                  float4 pack = {
                      half2float(row_x[col].x),
                      __half2float(row_x[col].y),
                      __half2float(row_x[col].z),
                      __half2float(row_x[col].w)};
                  buf[col] = pack.x;
                  buf[num_packs+col] = pack.y;
                  buf[2*num_packs+col] = pack.z;
                  buf[3*num_packs+col] = pack.w;

                  local_max[0] = max(local_max[0], max(max(pack.x, pack.y), max(pack.z, pack.w)));
              }
              blockReduceMax<float, 1>(local_max);//reduce on a block of #blockDim.x

              __shared float s_max;
              if (threadIdx.x == 0) {
                  s_max = local_max[0];
              }
              __syncthreads();

              float local_sum[1] = {0.0f};
              for (int i = tid; i < n; i += blockDim.x) {
                  float local_val = exp(buf[i]-s_max);
                  buf[i] = local_val;
                  local_sum[0] += local_val;
              }
              blockReduceSum<float, 1>(local_sum);

              __shared float s_sum;
              if (threadIdx.x == 0) {
                  s_sum = local_sum[0];
              }
              __syncthreads();

              for (int i = tid; i < num_packs; i += blockDim.x) {
                  const int col = i;
                  row_y[col] = {
                      __float2half_rn(buf[i]/s_sum),
                      __float2half_rn(buf[num_packs+i]/s_sum),
                      __float2half_rn(buf[2*num_packs+i]/s_sum),
                      __float2half_rn(buf[3*num_packs+i]/s_sum)};
              }
          }
      }


}
