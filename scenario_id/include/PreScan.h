//
// Created by oke on 17.01.19.
//

#ifndef TRAFFIC_SIM_PRESCAN_H
#define TRAFFIC_SIM_PRESCAN_H

#include <stddef.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#define CUDA_GLOB __global__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#define CUDA_GLOB
#endif


void CalculatePreSum(size_t *out, size_t out_size, size_t *in, int size, int batch_count);
int GetRequiredPreSumReqBufferSize(int size, int batch_count);

CUDA_DEV void PreScan(size_t *temp, size_t idx, size_t n, size_t skip=1);

CUDA_GLOB void GeneralPreSumKernel(size_t *out, size_t out_len, size_t *in, size_t in_len, int skip, int offset);
CUDA_GLOB void GeneralPreSumMergeKernel(size_t *out, size_t out_len, size_t *in, size_t in_len, size_t block_size);

#endif //TRAFFIC_SIM_PRESCAN_H
