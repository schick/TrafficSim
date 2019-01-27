//
// Created by oke on 17.01.19.
//

#include "PreScan.h"
#include "cuda/cuda_utils.h"
#include <assert.h>
#include <vector>

size_t GetRequiredPreSumReqBufferSize(size_t size, size_t batch_count) {
    size_t total_size = 0;
    for(size_t step_size = size; step_size >= 1; step_size = step_size == 1 ? 0 : step_size / batch_count + 1) {
        total_size += step_size;
    }
    return total_size;
}

__global__ void TestPreSum(size_t *out, size_t *in, int size) {
    if (GetGlobalIdx() == 0) printf("Test presum...\n");
    if(GetGlobalIdx() < size && GetGlobalIdx() > 0) {
        assert(out[GetGlobalIdx()] - out[GetGlobalIdx() - 1] == in[GetGlobalIdx()]);
    }
    if(GetGlobalIdx() == 0) {
        assert(out[GetGlobalIdx()] == in[GetGlobalIdx()]);
    }

}


void CalculatePreSum(size_t *out, size_t out_size, size_t *in, int size, int batch_count) {
    if(size == 0) return;
    assert(IsPowerOfTwo(batch_count));
    assert(out_size >= GetRequiredPreSumReqBufferSize(size, batch_count));
    int offset = 0;
    size_t *tmp_in = in;
    std::vector<int > sizes;
    int step_size;

    int input_array_size = size;
    for(step_size = size; step_size >= 1; step_size = step_size == 1 ? 0 : step_size / batch_count + 1) {
        sizes.push_back(step_size);

        GeneralPreSumKernel<<<step_size / batch_count + 1, batch_count / 2, batch_count * sizeof(size_t)>>>(
                out + offset, step_size, tmp_in, input_array_size,
                        step_size == size ? 1 : batch_count, step_size == size ? 0 : batch_count - 1);
        CHECK_FOR_ERROR();

        tmp_in = out + offset;
        offset += step_size;
        input_array_size = step_size;
    }
    step_size = sizes.back();
    sizes.pop_back();
    while(step_size < size) {
        offset -= step_size;
        GeneralPreSumMergeKernel<<<SUGGESTED_THREADS / 2, SUGGESTED_THREADS>>>(
                out + offset - sizes.back(), sizes.back(), out + offset, step_size, batch_count);
        CHECK_FOR_ERROR();

        step_size = sizes.back();
        sizes.pop_back();
    }
    // offset -= step_size;
    // size_t * res = out + offset;
#ifdef RUN_WITH_TESTS
    TestPreSum<<<size / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>(out, in, size);
    CHECK_FOR_ERROR();
#endif
}


__device__ void PreScan(size_t *temp, size_t idx, size_t n, size_t skip) {

    assert(IsPowerOfTwo(n));
    assert(IsPowerOfTwo(skip));
    if (!(2 * idx < n)) printf("%lu, %lu\n", 2*idx, n);
    assert(2 * idx < n);

    int offset = 1;

    n /= skip;

    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();

        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;

            temp[bi] += temp[ai * skip];
        }
        offset *= 2;
    }
    size_t total_sum = 0;
    if (idx == 0) {
        total_sum = temp[n - 1];
        temp[n - 1] = 0;
    } // clear the last element

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();

        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            //printf("%d, %d, %d, %d\n", ai, bi, offset, idx);

            size_t t = temp[ai * skip];
            temp[ai * skip] = temp[bi * skip];
            temp[bi * skip] += t;
        }
    }

    __syncthreads();
    size_t t1 = temp[2 * idx];
    size_t t2 = temp[2 * idx + 1];
    __syncthreads();

    if (idx != 0)
        temp[2 * idx - 1] = t1;
    else
        temp[n - 1] = total_sum;
    temp[2 * idx] = t2;
}


__global__ void GeneralPreSumKernel(size_t *out, size_t out_len, size_t *in, size_t in_len, int skip, int offset) {
    // if (GetGlobalIdx() < 5) printf("(%lu, %p), (%lu, %p)\n", out_len, out, in_len, in);

    size_t idx = GetThreadIdx();

    extern __shared__ size_t shared_presize[];
    size_t tmp_idx1 = (2 * idx    );
    size_t tmp_idx2 = (2 * idx + 1);

    size_t out_idx1 = tmp_idx1 + GetBlockIdx() * GetBlockDim() * 2;
    size_t out_idx2 = tmp_idx2 + GetBlockIdx() * GetBlockDim() * 2;

    size_t in_idx1 = out_idx1 * skip + offset;
    size_t in_idx2 = out_idx2 * skip + offset;

    if(in_idx1 < in_len) shared_presize[tmp_idx1] = in[in_idx1]; else shared_presize[tmp_idx1] = 0;
    if(in_idx2 < in_len) shared_presize[tmp_idx2] = in[in_idx2]; else shared_presize[tmp_idx2] = 0;

    PreScan(shared_presize, idx, GetBlockDim() * 2);

    __syncthreads();
    if(out_idx1 < out_len) out[out_idx1] = shared_presize[tmp_idx1];
    if(out_idx2 < out_len) out[out_idx2] = shared_presize[tmp_idx2];
}

__global__ void GeneralPreSumMergeKernel(size_t *out, size_t out_len, size_t *in, size_t in_len, size_t block_size) {
    for(size_t i=GetGlobalIdx(); i < out_len; i += GetGlobalDim()) {
        if(i >= block_size)
            out[i] += in[((i) / block_size) - 1];
    }
}
