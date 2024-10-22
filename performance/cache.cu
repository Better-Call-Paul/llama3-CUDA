#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Check CUDA errors
#define CUDA_CHECK_ERROR(call)                                           \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__,     \
                    __LINE__, cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// Number of iterations each thread will perform
#define NUM_ITERATIONS 1000000

// Kernel to perform memory read operations
__global__ void read_kernel(int *d_data, int num_iterations, volatile int *d_sink) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int temp = 0;
    
    for (int i = 0; i < num_iterations; ++i) {
        temp += d_data[idx % 1024]; // Modulo to ensure access within bounds
    }
    
    d_sink[idx] = temp; // Prevent compiler optimization
}

int main(int argc, char *argv[]) {
    // Parameters
    int max_threads = 1024; // Maximum threads per block (adjust based on GPU architecture)
    int num_blocks = 128;   // Number of blocks to launch
    size_t data_size = 1024 * sizeof(int); // Size of data each block will access

    // Allocate host memory
    int *h_data = (int *)malloc(data_size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    // Initialize host data
    for (int i = 0; i < 1024; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_data, data_size));
    CUDA_CHECK_ERROR(cudaMemcpy(d_data, h_data, data_size, cudaMemcpyHostToDevice));

    // Allocate sink to prevent compiler optimization
    volatile int *d_sink;
    CUDA_CHECK_ERROR(cudaMalloc((void **)&d_sink, max_threads * num_blocks * sizeof(int)));
    CUDA_CHECK_ERROR(cudaMemset((void *)d_sink, 0, max_threads * num_blocks * sizeof(int)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    printf("Measuring read queue FIFO depth...\n");
    printf("Threads\tTime (ms)\n");

    // Iterate over different numbers of threads (powers of two)
    for (int threads = 32; threads <= max_threads; threads *= 2) {
        dim3 block(threads);
        dim3 grid(num_blocks);

        // Record the start event
        CUDA_CHECK_ERROR(cudaEventRecord(start, 0));

        // Launch the kernel
        read_kernel<<<grid, block>>>(d_data, NUM_ITERATIONS, d_sink);

        // Record the stop event
        CUDA_CHECK_ERROR(cudaEventRecord(stop, 0));

        // Wait for the stop event to complete
        CUDA_CHECK_ERROR(cudaEventSynchronize(stop));

        // Calculate elapsed time
        float elapsed_time = 0.0f;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&elapsed_time, start, stop));

        printf("%d\t%.3f\n", threads, elapsed_time);

        // Optionally, add a small delay to ensure accurate measurements
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    }

    // Clean up
    CUDA_CHECK_ERROR(cudaEventDestroy(start));
    CUDA_CHECK_ERROR(cudaEventDestroy(stop));
    CUDA_CHECK_ERROR(cudaFree(d_data));
    CUDA_CHECK_ERROR(cudaFree((void *)d_sink));
    free(h_data);

    return 0;
}
