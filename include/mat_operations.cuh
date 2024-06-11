#ifndef MAT_OPERATIONS_CUH
#define MAT_OPERATIONS_CUH


namespace llama {

template<typename T>
__global__ void rms_norm(const T* input, const T* output) {

}

template<typename T>
__global__ void softmax(const T* a, const T* b, const T* c, int N, int M) {

}


/*
 * a: N x M dim
 * b: M x P dim
*/
template<typename T, int N, int M, int P>
__global__ void mat_mul(const T* a, const T* b, T* c) {
    int r = blockDim.x * blockIdx.x + threadIdx.x, c = blockDim.y * blockIdx.y + threadIdx.y;
    if (r < M && c < P) {
        T sum = 0;
        #pragma unroll
        for (int i = 0; i < M; i++) {
            sum += a[r * M + i] * b[i * P + c];
        }
        c[r * P + col] = sum;
    }
}



}



#endif