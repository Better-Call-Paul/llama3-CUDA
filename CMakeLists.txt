cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(llama3 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# It's better to use `target_include_directories` for specific targets rather than globally setting it.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDA REQUIRED)

# Add CUTLASS directory. You must adjust this path to where CUTLASS is located on your system.
set(CUTLASS_DIR "/path/to/cutlass")
include_directories(${CUTLASS_DIR}/include)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_60)

add_executable(LLAMA3_CUDA src/main.cu)

# Using `set_property` to specify CUDA architectures is good.
set_property(TARGET LLAMA3_CUDA PROPERTY CUDA_ARCHITECTURES 60 61 70 75 80 86)

# Ensure that the target name in `target_link_libraries` matches the target you define in `add_executable`
target_link_libraries(LLAMA3_CUDA ${CUDA_LIBRARIES})

# Optionally, you could link CUTLASS as needed (if it provides a library target, otherwise it's header-only)
# target_link_libraries(LLAMA3_CUDA cutlass)
