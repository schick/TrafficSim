
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS -O3)

# find_package(CUDA QUIET REQUIRED)
# Pass options to NVCC
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
#        -O3)

add_executable(traffic_sim main.cpp)