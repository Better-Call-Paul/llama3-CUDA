#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>  // For malloc and free
#include <cmath>    // For fabs

// CUDA error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Matrix dimensions (can be adjusted as needed)
const int M = 1024; // Number of rows in A and C
const int N = 1024; // Number of columns in B and C
const int K = 1024; // Number of columns in A and rows in B

// CUDA kernel for matrix multiplication (C = A * B)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Calculate the row index of the C matrix
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of the C matrix
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float value = 0.0f;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    // Size calculations
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory using malloc
    float* h_A = (float*)malloc(size_A);
    float* h_B = (float*)malloc(size_B);
    float* h_C = (float*)malloc(size_C);

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize host matrices (for example, fill with 1.0f)
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    // Using 16x16 threads per block for good occupancy
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y,
                 1);

    // Launch the matrix multiplication kernel
    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Check for any kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate the elapsed time between the two events
    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    std::cout << "Matrix multiplication took " << elapsedTime << " ms" << std::endl;

    // Copy the result back to the host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));


    std::cout << "Performing CPU matrix multiplication for verification..." << std::endl;
    float* cpu_C = (float*)malloc(size_C);
    if (cpu_C == nullptr) {
        std::cerr << "Failed to allocate CPU result matrix." << std::endl;
        free(h_A);
        free(h_B);
        free(h_C);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        exit(EXIT_FAILURE);
    }

    // Initialize CPU result matrix
    for (int i = 0; i < M * N; ++i) {
        cpu_C[i] = 0.0f;
    }

    // Perform CPU matrix multiplication
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int e = 0; e < K; ++e) {
                sum += h_A[i * K + e] * h_B[e * N + j];
            }
            cpu_C[i * N + j] = sum;
        }
    }

    // Verify the results
    bool correct = true;
    const float epsilon = 1e-3f; // Tolerance for floating-point comparison
    for (int i = 0; i < M * N; ++i) {
        if (fabs(cpu_C[i] - h_C[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": CPU " << cpu_C[i] << " vs GPU " << h_C[i] << std::endl;
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "Verification passed: GPU results match CPU results within tolerance." << std::endl;
    } else {
        std::cout << "Verification failed: GPU results do not match CPU results." << std::endl;
    }

    // Free CPU verification memory
    free(cpu_C);
    

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
