#include <iostream>

using namespace std;

#define SIZE 256 
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction(int *vec, int* vec_result, clock_t* time) {
    // issue, earlier thread blocks were doing more work
    // number of threads is decreasing by half
    // so idle time gap grows each iteration

    if (threadIdx.x == 0) {
        time[blockIdx.x] = clock();
    }

    __shared__ int partial_sum[SHMEM_SIZE];

    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = vec[global_index];

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // first one will write to mem
    if (threadIdx.x == 0) {
        vec_result[blockIdx.x] = partial_sum[0];
    }
    if (threadIdx.x == 0) {
        time[blockIdx.x + blockDim.x] = clock();
    }

}

void initialize_vector(int* v, int n) {
    for (int i = 0; i < n; ++i) {
        v[i] = i;
    }
}


int main() {
    int N = 1 << 16;
    size_t bytes = N * sizeof(int);
    int *host_vector, *host_vector_result;
    int *device_vector, *device_vector_result;

    host_vector = (int*)malloc(bytes);
    host_vector_result = (int*)malloc(bytes);
    cudaMalloc(&device_vector, bytes);
    cudaMalloc(&device_vector_result, bytes);

    initialize_vector(host_vector, N);

    cudaMemcpy(device_vector, host_vector, bytes, cudaMemcpyHostToDevice);

    int THREADS_PER_BLOCK = SIZE;

    int GRID_SIZE = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    clock_t* host_clock = new clock_t[GRID_SIZE * 2];
    clock_t* device_clock;
    cudaMalloc(&device_clock, GRID_SIZE * 2 * sizeof(clock_t));

    sum_reduction<<<GRID_SIZE, THREADS_PER_BLOCK>>>(device_vector, device_vector_result, device_clock);

    cudaMemcpy(host_clock, device_clock, GRID_SIZE * 2 * sizeof(clock_t), cudaMemcpyDeviceToHost);

    sum_reduction<<<1, THREADS_PER_BLOCK>>>(device_vector, device_vector_result, device_clock);

    cudaMemcpy(host_vector_result, device_vector_result, bytes, cudaMemcpyDeviceToHost);


    cout << "Block, Clocks" << "\n";
    for (int i = 0; i < GRID_SIZE; ++i) {
        cout << i << " , " << host_clock[i + GRID_SIZE] - host_clock[i] << "\n";
    }

    // allocate host and device 

    // populate on host and memcpy to device

    // setup threads and block size

    // kernel launch

    // prine time values 
    cudaFree(device_vector);
    cudaFree(device_vector_result);
    cudaFree(device_clock);

    return 0;
}