cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(llama3_cuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Adjust these to match your target GPUs

# Find CUDA package
find_package(CUDA REQUIRED)

# Add CUDA include directories
include_directories(${CUDA_INCLUDE_DIRS})

# Set source files
set(SOURCES
    src/main.cu
    src/tokenizer.cu
    src/attention.cu
    src/feedforward.cu
    src/layernorm.cu
)

# Create the executable
add_executable(llama3_cuda ${SOURCES})

# Link against CUDA libraries
target_link_libraries(llama3_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

# Set CUDA specific flags
set_target_properties(llama3_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add include directories
target_include_directories(llama3_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set compiler flags
if(MSVC)
    target_compile_options(llama3_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
else()
    target_compile_options(llama3_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall -Wextra -Werror>)
endif()

# Enable debug symbols in Debug mode
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")

# Optimize for speed in Release mode
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")