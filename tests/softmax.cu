#include <cuda_runtime.h>
#include <../include/softmax.cuh>

// Function to initialize input data with random values
void initialize_data(std::vector<float>& data) {
    for (auto& x : data) {
        x = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Function to compare two vectors and check if they are approximately equal
bool compare_results(const std::vector<float>& a, const std::vector<float>& b, float epsilon = 1e-5) {
    if (a.size() != b.size()) {
        std::cerr << "Result size mismatch!" << std::endl;
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (fabs(a[i] - b[i]) > epsilon) {
            std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Function to perform Softmax using cuDNN
void cudnn_softmax(cudnnHandle_t cudnn,
                  const float* d_input,
                  float* d_output,
                  int num_rows,
                  int num_cols) {
    // Create tensor descriptor
    cudnnTensorDescriptor_t tensor_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&tensor_desc));

    // Set tensor dimensions (N x C x 1 x 1)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(tensor_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          num_rows,    // N
                                          num_cols,    // C
                                          1,           // H
                                          1));         // W

    // Perform Softmax
    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnSoftmaxForward(cudnn,
                                    CUDNN_SOFTMAX_ACCURATE,
                                    CUDNN_SOFTMAX_MODE_CHANNEL,
                                    &alpha,
                                    tensor_desc,
                                    d_input,
                                    &beta,
                                    tensor_desc,
                                    d_output));

    // Destroy tensor descriptor
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(tensor_desc));
}
