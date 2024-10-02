cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(llama3 C CXX CUDA)

set(CMAKE_CXX_STANDARD 11)  
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

add_executable(LLAMA3_CUDA src/main.cu src/bpe_tokenizer.cu 
src/mat_mul.cu src/multi_head_attention.cu src/rms_norm.cu 
src/rope_rotation.cu src/softmax.cu src/transformer.cu src/utils.cu)

set_property(TARGET LLAMA3_CUDA PROPERTY CUDA_ARCHITECTURES 60 61 70 75 80 86)

target_link_libraries(llama3 ${CUDA_LIBRARIES})