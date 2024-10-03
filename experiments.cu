#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

// Define the size of the data array (256 MB)
#define ARRAY_SIZE (256 * 1024 * 1024 / sizeof(float))
// Define the number of threads and blocks
#define THREADS_PER_BLOCK 256
#define BLOCKS 64

// CUDA kernel that performs N independent memory reads per thread
__global__ void memory_read_kernel(float *data, int N, float *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float tmp = 0.0f;

    // Each thread reads from different locations to avoid cache effects
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        // Calculate a unique index to read from
        int index = (idx * N + i * 123457) % ARRAY_SIZE;
        tmp += data[index];
    }

    // Write the result to the output array
    output[idx] = tmp;
}

int main() {
    // Calculate the total number of threads
    size_t total_threads = THREADS_PER_BLOCK * BLOCKS;
    // Calculate the size of the data and output arrays
    size_t data_size = ARRAY_SIZE * sizeof(float);
    size_t output_size = total_threads * sizeof(float);

    // Allocate host memory
    float *h_data = new float[ARRAY_SIZE];
    float *h_output = new float[total_threads];

    // Allocate device memory
    float *d_data, *d_output;
    cudaMalloc(&d_data, data_size);
    cudaMalloc(&d_output, output_size);

    // Initialize the data array with some values
    for (size_t i = 0; i < ARRAY_SIZE; ++i) {
        h_data[i] = static_cast<float>(i % 1000);
    }

    // Copy the data array from host to device
    cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice);

    std::cout << "Measuring read queue FIFO depth..." << "\n";

    // Loop over different values of N (number of memory reads per thread)
    for (int N = 1; N <= 64; N *= 2) {
        // warm up by launching kernel
        memory_read_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, N, d_output);
        cudaDeviceSynchronize();
        // start clock
        auto start = std::chrono::high_resolution_clock::now();
        // launch kernel
        memory_read_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_data, N, d_output);
        cudaDeviceSynchronize();
        // end clock
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_spent = end - start;
        // print out time diff 
        std::cout << "N = " << N << ", Time = " << time_spent.count() << " s" << "\n";
    }

    // Clean up memory
    delete[] h_data;
    delete[] h_output;
    cudaFree(d_data);
    cudaFree(d_output);

    return 0;
}
