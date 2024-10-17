cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(llama3 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

file(GLOB_RECURSE SOURCES "src/*.cu")

add_executable(LLAMA3_CUDA ${SOURCES})

set_property(TARGET LLAMA3_CUDA PROPERTY CUDA_ARCHITECTURES 60 61 70 75 80 86)

target_link_libraries(LLAMA3_CUDA ${CUDA_LIBRARIES})

# target_link_libraries(LLAMA3_CUDA cutlass)

set_target_properties(LLAMA3_CUDA PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")