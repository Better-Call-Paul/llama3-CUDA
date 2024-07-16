cmake_minimum_required(VERSION 2.8)

project(llama3 C CXX)

find_package(CUDA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

cuda_add_executable(llama3 src/transformer.cu src/utils.cu 
src/main.cu src/bpe_tokenizer.cu src/mat_mul.cu src/softmax.cu src/multi_head_attention.cu src/rms_norm.cu
src/rope_rotation.cu)

target_link_libraries(llama3 ${CUDA_LIBRARIES})