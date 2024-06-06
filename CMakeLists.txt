cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(llama3_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


#[[ Set compute for target GPU ]]
# set(CMAKE_CUDA_ARCHITECTURES XX)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(llama3_cuda src/main.cu)


