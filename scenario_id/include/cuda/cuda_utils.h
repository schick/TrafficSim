//
// Created by oke on 23.12.18.
//


/**
 * Inplace bitonic sort using CUDA.
 */

#ifndef CUDA_UTILS
#define CUDA_UTILS
#include <exception>
#include <stdexcept>

#ifdef DEBUG
#define CHECK_FOR_ERROR() { cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());}
#else
#define CHECK_FOR_ERROR() { gpuErrchk(cudaPeekAtLastError()); }
// #define CHECK_FOR_ERROR() { cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());}
#endif


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

#define SUGGESTED_THREADS 512

#define PRE_SUM_BLOCK_SIZE 512



extern size_t total_gpu_alloc;
#ifdef DEBUG_MSGS
#define GPU_ALLOC(B, S) \
    gpuErrchk(cudaMalloc(B, S)); \
    total_gpu_alloc += S; \
    printf("Allocated: %7.2f, Total: %7.2f in %s:%d\n", (double) S / 1024. / 1024., (double) total_gpu_alloc / 1024. / 1024., __FILE__, __LINE__);
#else
#define GPU_ALLOC(B, S) \
    gpuErrchk(cudaMalloc(B, S));
#endif

#define CUDA_GLOBAL_ITER(VAR_NAME, MAX) size_t each = (MAX) / GetGlobalDim() + 1; \
    size_t begin = each * GetGlobalIdx(); \
    for(size_t VAR_NAME = begin; VAR_NAME < begin + each && VAR_NAME < (MAX); VAR_NAME++)
#define CUDA_THREAD_ITER(VAR_NAME, MAX) size_t each = (MAX) / GetBlockDim() + 1; \
    size_t begin = each * GetThreadIdx(); \
    for(size_t VAR_NAME = begin; VAR_NAME < begin + each && VAR_NAME < (MAX); VAR_NAME++)
#define CUDA_BLOCK_ITER(VAR_NAME, MAX) size_t each = (MAX) / GetGridDim() + 1; \
    size_t begin = each * GetBlockIdx(); \
    for(size_t VAR_NAME = begin; VAR_NAME < begin + each && VAR_NAME < (MAX); VAR_NAME++)
#define CUDA_GLOBAL_ITER2(NAME, MAX) for(size_t NAME=GetGlobalIdx(); NAME < MAX; NAME += GetGlobalDim())


template<typename T, typename Cmp>
CUDA_HOSTDEV inline size_t *lower_bound(T *__first, T *__last, const T& __val, Cmp __comp)
{
    size_t __len = __last - __first;

    while (__len > 0)
    {
        size_t __half = __len >> 1;
        size_t __middle = __first + __half;
        if (__comp(__middle, __val))
        {
            __first = __middle;
            ++__first;
            __len = __len - __half - 1;
        }
        else
            __len = __half;
    }
    return __first;
}

template<typename T>
CUDA_HOSTDEV inline size_t *lower_bound(T *__first, T *__last, const T& __val)
{
    size_t __len = __last - __first;

    while (__len > 0)
    {
        size_t __half = __len >> 1;
        size_t *__middle = __first + __half;
        if (*__middle < __val)
        {
            __first = __middle;
            ++__first;
            __len = __len - __half - 1;
        }
        else
            __len = __half;
    }
    return __first;
}



template<typename T, typename Cmp>
CUDA_HOSTDEV inline T *upper_bound(T *__first, T *__last, const T& __val, Cmp __comp) {
    size_t __len = __last - __first;

    while (__len > 0)
    {
        size_t __half = __len >> 1;
        T *__middle = __first + __half;
        if (__comp(__val, *__middle))
            __len = __half;
        else
        {
            __first = __middle;
            ++__first;
            __len = __len - __half - 1;
        }
    }
    return __first;
}


template<typename T>
CUDA_HOSTDEV inline T *upper_bound(T *__first, T *__last, const T& __val) {
    size_t __len = __last - __first;

    while (__len > 0)
    {
        size_t __half = __len >> 1;
        T *__middle = __first + __half;
        if (__val < *__middle)
            __len = __half;
        else
        {
            __first = __middle;
            ++__first;
            __len = __len - __half - 1;
        }
    }
    return __first;
}


template<typename T>
CUDA_HOSTDEV inline void cu_swap(T &t1, T &t2) {
    T t = t1;
    t1 = t2;
    t2 = t;
}


CUDA_HOSTDEV inline uint64_t next_pow2m1(uint64_t x) {
    x--;
    x |= x>>1;
    x |= x>>2;
    x |= x>>4;
    x |= x>>8;
    x |= x>>16;
    x |= x>>32;
    x++;
    return x;
}


CUDA_HOSTDEV inline bool IsPowerOfTwo(unsigned long x) {
    return (x != 0) && ((x & (x - 1)) == 0);
}

class CudaEx : std::runtime_error {
public:
    CudaEx(const std::string &c) : std::runtime_error(c) {}
};


CUDA_DEV size_t GetGlobalIdx();
CUDA_DEV size_t GetThreadIdx();
CUDA_DEV size_t GetBlockIdx();

CUDA_DEV size_t GetGridDim();
CUDA_DEV size_t GetBlockDim();
CUDA_DEV size_t GetGlobalDim();

#ifdef __CUDACC__

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {

        fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)  {
            throw CudaEx("Fail!");
        }
    }
    return 0;
}

#endif
#endif