//
// Created by oke on 17.01.19.
//

#include "cuda/cuda_utils.h"

CUDA_DEV
size_t GetGlobalIdx(){
    return + blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y
           + blockIdx.y * blockDim.x * blockDim.y * blockDim.z * gridDim.x
           + blockIdx.x * blockDim.x * blockDim.y * blockDim.z
           + threadIdx.z * blockDim.y * blockDim.x
           + threadIdx.y * blockDim.x
           + threadIdx.x;
}

CUDA_DEV
size_t GetThreadIdx(){
    return threadIdx.z * blockDim.y * blockDim.x
           + threadIdx.y * blockDim.x
           + threadIdx.x;
}
CUDA_DEV
size_t GetBlockIdx(){
    return blockIdx.z * gridDim.y * gridDim.x
           + blockIdx.y * gridDim.x
           + blockIdx.x;
}

CUDA_DEV
size_t GetGridDim(){
    return gridDim.y * gridDim.x * gridDim.z;
}

CUDA_DEV
size_t GetBlockDim(){
    return blockDim.y * blockDim.x * blockDim.z;
}

CUDA_DEV
size_t GetGlobalDim(){
    return blockDim.y * blockDim.x * blockDim.z * gridDim.y * gridDim.x * gridDim.z;
}

