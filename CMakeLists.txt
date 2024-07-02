cmake_minimum_required(VERSION 2.8)

project(llama3 C CXX)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

# Set CUDA flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

# Add CUDA executable
cuda_add_executable(llama3 src/main.cu src/bpe_tokenizer.cu src/mat_add.cu src/mat_mul.cu src/softmax.cu src/multi_head.cu src/flash_attention.cu src/layer_norm.cu
src/transformer.cu src/utils.cu)

# Link CUDA libraries
target_link_libraries(llama3 ${CUDA_LIBRARIES})