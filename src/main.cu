#include <iostream>

#define CUDA_ERROR_CHECKER(call) \
    do { \
        cudaError_t error = call \
        if (error != cudaSucess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error));
            exit(EXIT_FAILURE); \
        } \
    } while (0)


int main(int argc, char* argv[]) {

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    std::cout << "Stable" << "\n";
    
    return 0;
}