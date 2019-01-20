//
// Created by oke on 17.01.19.
//

#include <driver_types.h>
#include <algorithms/CudaAlgorithm.h>
#include "SortedBucketContainer.h"
#include "PreScan.h"

#define MAX(a, b) (a < b ? b : a)
#define MIN(a, b) (a < b ? a : b)

CUDA_HOSTDEV inline void GetBucketIdxFromGlobalIdx(size_t globalIdx, size_t *sizePrefixSum, size_t sizePrefixSumLen, size_t *bucket_idx, size_t *element_idx) {
    size_t *lb = upper_bound<size_t>(sizePrefixSum, sizePrefixSum + sizePrefixSumLen, globalIdx);
    *element_idx = lb == sizePrefixSum ? globalIdx : globalIdx - *(lb - 1);
    *bucket_idx = lb - sizePrefixSum;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ---------------------------- SORTING --------------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

unsigned int size = 2;

CUDA_GLOB void BitonicSortMergeKernel(BucketData *buckets, size_t num_buckets, size_t *sizePrefixSum, size_t sizePrefixSumLen, int k, TrafficObject_id::Cmp cmp) {
    size_t element_idx, bucket_idx;
    GetBucketIdxFromGlobalIdx(GetGlobalIdx(), sizePrefixSum, sizePrefixSumLen, &bucket_idx, &element_idx);

    if (bucket_idx == num_buckets) return;
    assert(bucket_idx < num_buckets);

    auto &bucket = buckets[bucket_idx].buffer;
    auto n = buckets[bucket_idx].size;
    assert(element_idx < n);

    unsigned int i = element_idx;
    if(i + k < n && cmp(bucket[i + k], bucket[i]))
        cu_swap(bucket[i], bucket[i + k]);


#ifdef RUN_WITH_TESTS
    if(element_idx == 0 && buckets[bucket_idx].id == BUCKET_TO_ANALYZE) {
        printf("Bucket(%lu) of len(%lu) after sorting step: ", bucket_idx, buckets[bucket_idx].size);
        for (int i = 0; i < buckets[bucket_idx].size; i++) {
            TrafficObject_id *p_obj = bucket[i];
            printf(" %lu(%.2f), ", p_obj == nullptr ? (size_t )-1 : p_obj->id, p_obj == nullptr ? -1. : p_obj->x);
        }
        printf("\n");
    }
#endif
}

CUDA_GLOB void BitonicSortStepKernel(BucketData *buckets, size_t num_buckets, size_t *sizePrefixSum, size_t sizePrefixSumLen, int j, int k, TrafficObject_id::Cmp cmp) {
    size_t element_idx, bucket_idx;
    GetBucketIdxFromGlobalIdx(GetGlobalIdx(), sizePrefixSum, sizePrefixSumLen, &bucket_idx, &element_idx);

    if (bucket_idx == num_buckets) return;
    assert(bucket_idx < num_buckets);

    auto &bucket = buckets[bucket_idx].buffer;
    auto n = buckets[bucket_idx].size;
    assert(element_idx < n);

    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = element_idx;
    ixj = i ^ j;

    if (i >= n || ixj >= n)
        return;
    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (cmp(bucket[ixj], bucket[i])) {
                /* exchange(i,ixj); */
                cu_swap(bucket[i], bucket[ixj]);
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (cmp(bucket[i], bucket[ixj])) {
                /* exchange(i,ixj); */
                cu_swap(bucket[i], bucket[ixj]);
            }
        }
    }
}

template<typename T, typename Cmp>
__device__ void bitonic_sort_step(T *dev_values, int j, int k, int n, Cmp cmp) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * threadIdx.y;
    ixj = i ^ j;

    if (i >= n || ixj >= n)
        return;
    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (cmp(dev_values[ixj], dev_values[i])) {
                /* exchange(i,ixj); */
                cu_swap(dev_values[i], dev_values[ixj]);
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (cmp(dev_values[i], dev_values[ixj])) {
                /* exchange(i,ixj); */
                cu_swap(dev_values[i], dev_values[ixj]);
            }
        }
    }
}

template<typename T, typename Cmp>
__device__ void bitonic_sort_merge(T* values, int k, int n, Cmp cmp) {
    unsigned int i; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * threadIdx.y;
    if(i + k < n && cmp(values[i + k], values[i]))
        cu_swap(values[i], values[i + k]);
}

__device__ void cudaSort(BucketData *buckets, size_t bucket_count, TrafficObject_id::Cmp cmp, size_t length_from, size_t length_to, BucketData *largerBuckets, unsigned int *largerBucketsLastIdx) {

    size_t first_bucket_idx = GetBlockIdx();
    size_t i = GetThreadIdx();
    size_t block_count = GetGridDim();
    extern __shared__ TrafficObject_id *device_values[];
    assert(device_values != nullptr);
    for(size_t bucket_idx = first_bucket_idx; bucket_idx < bucket_count; bucket_idx += block_count) {
#ifdef RUN_WITH_TESTS
        if(i == 0 && buckets[bucket_idx].id == BUCKET_TO_ANALYZE) {
            printf("Bucket(%lu) contents before sorting: ", bucket_idx);
            for (int i = 0; i < buckets[bucket_idx].size; i++) {
                TrafficObject_id *p_obj = buckets[bucket_idx].buffer[i];
                printf(" %lu(%.2f), ", p_obj == nullptr ? (size_t )-1 : p_obj->id, p_obj == nullptr ? -1. : p_obj->x);
            }
            printf("\n");
        }
#endif
        if (length_from <= buckets[bucket_idx].size && buckets[bucket_idx].size <= length_to) {
            assert(buckets[bucket_idx].size <= GetBlockDim());

            if (buckets[bucket_idx].size > 1) {
                TrafficObject_id **device_values_ = buckets[bucket_idx].buffer;
                size_t n = buckets[bucket_idx].size;
                // printf("bucket %lu with %lu items.\n", bucket_idx, n);
                if (GetThreadIdx() < n) {
                    device_values[GetThreadIdx()] = device_values_[GetThreadIdx()];
                }
                int j, k;
                unsigned long power = next_pow2m1(n);
                assert(power >= n);
                /* Major step */
                for (k = 2; k <= power; k <<= 1) {
                    /* Minor step */
                    for (j = k >> 1; j > 0; j = j >> 1) {
                        bitonic_sort_step(device_values, j, k, n, cmp);
                        __syncthreads();
                    }
                }

                for (unsigned long k = power; k > 0; k >>= 1) {
                    bitonic_sort_merge(device_values, k, n, cmp);
                    __syncthreads();
                }

                if (GetThreadIdx() < n)
                    device_values_[GetThreadIdx()] = device_values[GetThreadIdx()];


            }
        } else {
            if(GetThreadIdx() == 0)
                if(buckets[bucket_idx].size > length_to) {
                    // printf("isLarger: %lu (%lu, %lu)\n", buckets[bucket_idx].size, length_from, length_to);
                    largerBuckets[atomicAdd(largerBucketsLastIdx, 1)] = buckets[bucket_idx];
                } else{

                    //printf("lower: %lu (%lu, %lu)\n", buckets[bucket_idx].size, length_from, length_to);
                }
        }
#ifdef RUN_WITH_TESTS
        if(i == 0 && buckets[bucket_idx].id == BUCKET_TO_ANALYZE) {
            printf("Bucket(%lu) contents after sorting: ", bucket_idx);
            for (int i = 0; i < buckets[bucket_idx].size; i++) {
                TrafficObject_id *p_obj = buckets[bucket_idx].buffer[i];
                printf(" %lu(%.2f), ", p_obj == nullptr ? (size_t )-1 : p_obj->id, p_obj == nullptr ? -1. : p_obj->x);
            }
            printf("\n");
        }
#endif

    }
}

__global__ void cudaSortKernel(SortedBucketContainer *container, TrafficObject_id::Cmp  cmp, size_t length_from, size_t length_to, BucketData *largerBuckets, unsigned int *largerBucketsLastIdx) {
    BucketData *buckets = container->buckets;
    size_t bucket_count = container->bucket_count;
    cudaSort(buckets, bucket_count, cmp, length_from, length_to, largerBuckets, largerBucketsLastIdx);
}

__global__ void cudaSortKernel2(BucketData *buckets, unsigned int *bucket_count, TrafficObject_id::Cmp cmp, size_t length_from, size_t length_to, BucketData *largerBuckets, unsigned int *largerBucketsLastIdx) {
    cudaSort(buckets, *bucket_count, cmp, length_from, length_to, largerBuckets, largerBucketsLastIdx);
}

template <typename T>
__global__ void ReduceLargeSortInfo(T *max_size, T *num_cars, BucketData *buckets, size_t num_buckets) {
    size_t thread_idx = GetThreadIdx();
    size_t bucket_offset = GetBlockDim() * GetBlockIdx();

    size_t thread2;

    T temp;
    extern __shared__ T tmp_data[];
    T *max = tmp_data;
    T *sum = tmp_data + GetBlockDim();

    max[thread_idx] = (thread_idx + bucket_offset) > num_buckets ? 0 : buckets[thread_idx + bucket_offset].size;
    assert(max + thread_idx < sum);
    sum[thread_idx] = (thread_idx + bucket_offset) > num_buckets ? 0 : buckets[thread_idx + bucket_offset].size;
    size_t nTotalThreads = GetBlockDim(); // Total number of active threads
    __syncthreads();
    assert(IsPowerOfTwo(nTotalThreads));

    while (nTotalThreads > 1) {
        size_t halfPoint = (nTotalThreads >> 1); // divide by two
        assert(halfPoint * 2 == nTotalThreads);
        // only the first half of the threads will be active.
        if (thread_idx < halfPoint) {
            thread2 = thread_idx + halfPoint;

            temp = max[thread2];
            if (temp > max[thread_idx]) {
                max[thread_idx] = temp;
            }

            sum[thread_idx] += sum[thread2];
        }
        __syncthreads();

        // Reducing the binary tree size by two:
        nTotalThreads = halfPoint;
    }
    if (thread_idx == 0) {
        atomicAdd(num_cars, sum[0]);
        atomicMax(max_size, max[0]);
    }
}

using SizeType = unsigned int;

void SortedBucketContainer::SortLargeBuckets(BucketData *buckets, size_t num_buckets, Scenario_id &scenario, SortBuffer &sortBuffer) {

    SortedBucketContainer::FetchBucketSizes(buckets, num_buckets, scenario, sortBuffer.bucketSizes);
    CalculatePreSum(sortBuffer.laneBucketPreSumBuffer, sortBuffer.lanePreSumBufferSize, sortBuffer.bucketSizes, num_buckets, sortBuffer.preSumBatchSize);
    CHECK_FOR_ERROR();


    SizeType *dev_num_cars_n_max_bucket_len;
    gpuErrchk(cudaMalloc((void **) &dev_num_cars_n_max_bucket_len, 2 * sizeof(SizeType)));
    SizeType *dev_num_cars = dev_num_cars_n_max_bucket_len;
    SizeType *dev_max_bucket_len = dev_num_cars_n_max_bucket_len + 1;

    gpuErrchk(cudaMemset(dev_num_cars_n_max_bucket_len, 0, 2 * sizeof(SizeType)));

    ReduceLargeSortInfo<SizeType><<<num_buckets / SUGGESTED_THREADS + 1, SUGGESTED_THREADS, 2 * SUGGESTED_THREADS * sizeof(SizeType)>>>(dev_max_bucket_len, dev_num_cars, buckets, num_buckets);
    CHECK_FOR_ERROR();

    SizeType num_cars_n_max_bucket_len[2];
    SizeType &num_cars = num_cars_n_max_bucket_len[0];
    SizeType &max_bucket_len = num_cars_n_max_bucket_len[1];

    gpuErrchk(cudaMemcpy(num_cars_n_max_bucket_len, dev_num_cars_n_max_bucket_len, 2 * sizeof(SizeType), cudaMemcpyDeviceToHost));

    size_t n = next_pow2m1(max_bucket_len);
    dim3 blocks(num_cars / SUGGESTED_THREADS + 1);
    dim3 threads(MIN(num_cars, SUGGESTED_THREADS));

    assert(max_bucket_len <= 150);
    assert(num_cars <= scenario.cars.size());
    TrafficObject_id::Cmp cmp;
    int j, k;
    /* Major step */
    for (k = 2; k <= n; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            BitonicSortStepKernel<<<blocks, threads>>>(buckets, num_buckets, sortBuffer.laneBucketPreSumBuffer, num_buckets, j, k, cmp);
            CHECK_FOR_ERROR();
        }
    }

    unsigned long power = pow(2, floor(log(n)/log(2)));
    for (unsigned long k = power; k > 0; k >>= 1) {
        BitonicSortMergeKernel<<<blocks, threads>>>(buckets, num_buckets, sortBuffer.laneBucketPreSumBuffer, num_buckets, k, cmp);
        CHECK_FOR_ERROR();
    }

}

void SortedBucketContainer::SortInSizeSteps(SortedBucketContainer *container, Scenario_id &scenario, SortBuffer &sortBuffer) {

    TrafficObject_id::Cmp cmp;
    unsigned int my_size = size;
    cudaMemset(sortBuffer.last, 0, sizeof(unsigned int));
    cudaSortKernel<<<2048, my_size, my_size * sizeof(TrafficObject_id*)>>>(container, cmp, 0, my_size, sortBuffer.pBucketData, sortBuffer.last);
    CHECK_FOR_ERROR();

    unsigned int lastHost;
    gpuErrchk(cudaMemcpy(&lastHost, sortBuffer.last, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    while (lastHost != 0) {
        // printf("lastHost: %d\n", lastHost);
        my_size *= 2;
        if (my_size > 1024) {
            SortLargeBuckets(sortBuffer.pBucketData, lastHost, scenario, sortBuffer);
            break;
        }
        if ((float) lastHost / scenario.lanes.size() > 0.1) {
            size = my_size;
        }
        // printf("size: %u\n", my_size);
        cudaMemset(sortBuffer.last2, 0, sizeof(unsigned int));
        // printf("%lu, %p, %u, %p\n", lastHost, sortBuffer.pBucketData, my_size, sortBuffer.pBucketData2);
        cudaSortKernel2<<< lastHost, my_size, my_size * sizeof(TrafficObject_id*)>> >
                                              (sortBuffer.pBucketData, sortBuffer.last, cmp, 0, my_size, sortBuffer.pBucketData2, sortBuffer.last2);
        CHECK_FOR_ERROR();
        gpuErrchk(cudaMemcpy(&lastHost, sortBuffer.last2, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        cu_swap(sortBuffer.last2, sortBuffer.last);
        cu_swap(sortBuffer.pBucketData, sortBuffer.pBucketData2);
    }
}

void SortedBucketContainer::SortFixedSize(SortedBucketContainer *container, Scenario_id &scenario, SortBuffer &sortBuffer) {
    TrafficObject_id::Cmp cmp;
    cudaSortKernel<<<2048, 1024, 1024 * sizeof(TrafficObject_id*)>>>(container, cmp, 0, 100000000, sortBuffer.pBucketData, sortBuffer.last);
    CHECK_FOR_ERROR();
}


void SortedBucketContainer::Sort(SortedBucketContainer *container, Scenario_id &scenario, SortBuffer &sortBuffer) {
    SortInSizeSteps(container, scenario, sortBuffer);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ---------------------------- END SORTING ----------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 *  extract bucket sizes
 **/
__global__ void FetchBucketSizesKernel(SortedBucketContainer *container, size_t *bucketSizes) {
    for(size_t idx = GetGlobalIdx(); idx < container->bucket_count; idx += GetGlobalDim()) {
        bucketSizes[idx] = container->buckets[idx].size;
    }
}

__global__ void FetchBucketSizesPtrKernel(BucketData *buckets, size_t num_buckets, size_t *bucketSizes) {
    for(size_t idx = GetGlobalIdx(); idx < num_buckets; idx += GetGlobalDim()) {
        bucketSizes[idx] = buckets[idx].size;
    }
}

void SortedBucketContainer::FetchBucketSizes(SortedBucketContainer *container, Scenario_id &scenario, size_t *bucketSizes) {
    FetchBucketSizesKernel<<<sqrt(scenario.lanes.size()) + 1, sqrt(scenario.lanes.size()) + 1>>>(container, bucketSizes);
    CHECK_FOR_ERROR();
}

void SortedBucketContainer::FetchBucketSizes(BucketData *buckets, size_t num_buckets, Scenario_id &scenario, size_t *bucketSizes) {
    FetchBucketSizesPtrKernel<<<sqrt(num_buckets) + 1, sqrt(num_buckets) + 1>>>(buckets, num_buckets, bucketSizes);
    CHECK_FOR_ERROR();
}


__global__ void MergeBlockWisePreScan(size_t *out, size_t out_size, size_t *in, size_t in_size, size_t buffer_size, size_t skip_count) {
    assert(IsPowerOfTwo(buffer_size));
    size_t idx = GetThreadIdx();
    size_t offset = GetBlockIdx() * buffer_size * skip_count;

    assert(GetBlockDim() * 2 == buffer_size);
    assert((2 * GetBlockDim() + 1) * skip_count - 1 + GetGridDim() * buffer_size * skip_count >= in_size);

    extern __shared__ size_t temp[];

    if ((2 * idx + 1) * skip_count - 1 + offset < in_size)
        temp[2 * idx] = in[(2 * idx + 1) * skip_count - 1 + offset];
    else
        temp[2 * idx] = 0;

    if ((2 * idx + 2) * skip_count - 1 + offset < in_size)
        temp[2 * idx + 1] = in[(2 * idx + 2) * skip_count - 1 + offset];
    else
        temp[2 * idx + 1] = 0;

    __syncthreads();

    PreScan(temp, idx, buffer_size);

    __syncthreads();

    assert(2 * GetBlockDim() + 1 + buffer_size * GetGridDim() >= out_size);
    if(2 * idx + buffer_size * GetBlockIdx() < out_size)
        out[2 * idx + buffer_size * GetBlockIdx()] = temp[2 * idx];
    if(2 * idx + 1 + buffer_size * GetBlockIdx() < out_size)
        out[2 * idx + 1 + buffer_size * GetBlockIdx()] = temp[2 * idx + 1];
}

__global__ void MergeBlockWisePreScanStep2(size_t *out, size_t *in, size_t n, size_t buffer_size, size_t out_size) {
    size_t idx = GetThreadIdx();
    size_t offset = GetBlockIdx() * n;

    if (GetBlockIdx() > 0 && idx + offset < out_size) {
        out[idx + offset] += in[GetBlockIdx() - 1];
    }

}

__global__ void SetTempSizeKernel(SortedBucketContainer *container, unsigned int *temp_value) {
    size_t lane_id = GetGlobalIdx();

    assert(GetGlobalDim() >= container->bucket_count);

    if (lane_id < container->bucket_count) {
        container->buckets[lane_id].size += temp_value[lane_id];
#ifdef RUN_WITH_TESTS
        if (lane_id == BUCKET_TO_ANALYZE)
            printf("New Temporary Bucket(%lu) size (added %u): %lu\n", (size_t) BUCKET_TO_ANALYZE,temp_value[lane_id],
                   container->buckets[lane_id].size);
#endif
    }
}

struct free_deleter
{
    void operator()(void* m) {
        SortedBucketContainer *memory = (SortedBucketContainer *) m;
        SortedBucketContainer memory_host;

        cudaMemcpy(&memory_host, memory, sizeof(SortedBucketContainer), cudaMemcpyDeviceToHost);

        cudaFree(memory_host.main_buffer);
        cudaFree(memory_host.buckets);
        cudaFree(memory);
    }
};

__global__ void FixSizeKernel(SortedBucketContainer *container, bool only_lower) {
    for (size_t lane_id = GetBlockIdx(); lane_id < container->bucket_count; lane_id += GetGridDim()) {
        auto &bucket = container->buckets[lane_id];
        bool found = false;
        size_t new_size;
        size_t max_size = only_lower ? bucket.size : bucket.buffer_size;
        for (size_t idx = GetThreadIdx(); idx < max_size; idx += GetBlockDim()) {
            if (idx == 0) {
                if (bucket.buffer[0] == nullptr) {
                    new_size = 0;
                    found = true;
                }
            } else {
                if (bucket.buffer[idx] == nullptr && bucket.buffer[idx - 1] != nullptr) {
                    new_size = idx;
                    found = true;
                }
            }
        }

        if (found) bucket.size = new_size;

#ifdef RUN_WITH_TESTS
        if (found) {
            if(lane_id == BUCKET_TO_ANALYZE)
                printf("Final Bucket(%lu) size: %lu\n", lane_id, new_size);
        }
#endif
    }
}


__global__ void FixSizeKernel2(SortedBucketContainer *container, size_t *lanePreSum, size_t lanePreSumSize, size_t n) {
    for (size_t idx = GetGlobalIdx(); idx < 2 * n; idx += GetGlobalDim()) {
        size_t element_idx, bucket_idx;
        GetBucketIdxFromGlobalIdx(idx, lanePreSum, lanePreSumSize, &bucket_idx, &element_idx);
        if (bucket_idx >= container->bucket_count) return;
#ifdef DEBUG_MSGS
        if (element_idx >= container->buckets[bucket_idx].size)
            printf("ERROR: %lu: %lu, %lu, %lu, %lu\n", idx, bucket_idx, container->bucket_count, element_idx,
                   container->buckets[bucket_idx].size);
#endif
        assert(bucket_idx < container->bucket_count);
        assert(element_idx < container->buckets[bucket_idx].size);
        bool found = false;
        size_t new_size;
        auto &bucket = container->buckets[bucket_idx];
        if (element_idx == 0) {
            if (bucket.buffer[0] == nullptr) {
                new_size = 0;
                found = true;
            }
        } else {
            if (bucket.buffer[element_idx] == nullptr && bucket.buffer[element_idx - 1] != nullptr) {
                new_size = element_idx;
                found = true;
            }
        }

        if (found) {
            // printf("found size for(%lu): %lu -> %lu\n", bucket_idx, bucket.size, new_size);
            bucket.size = new_size;
        } else {
            // printf("%lu, %lu: %lu\n", bucket_idx, element_idx, bucket.size);
        }
    }
}
CUDA_HOST void SortedBucketContainer::FixSize(SortedBucketContainer *container, Scenario_id &scenario, bool only_lower, SortBuffer &sortBuffer) {
    SortedBucketContainer::FetchBucketSizes(container, scenario, sortBuffer.bucketSizes);
    CalculatePreSum(sortBuffer.laneBucketPreSumBuffer, sortBuffer.lanePreSumBufferSize, sortBuffer.bucketSizes, scenario.lanes.size(), sortBuffer.preSumBatchSize);
    CHECK_FOR_ERROR();

    FixSizeKernel2<<<scenario.cars.size() / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>(container, sortBuffer.laneBucketPreSumBuffer, scenario.lanes.size(), scenario.cars.size());
    CHECK_FOR_ERROR()

    // FixSizeKernel<<<256, 20>>>(bucketMemory, only_lower);
    // CHECK_FOR_ERROR();
}

CUDA_HOST void SortedBucketContainer::FixSize(SortedBucketContainer *container, bool only_lower) {
    FixSizeKernel<<<SUGGESTED_THREADS, SUGGESTED_THREADS>>>(container, only_lower);
    CHECK_FOR_ERROR();
}

CUDA_GLOB void bucketMemoryLoadKernel2(SortedBucketContainer *bucketmem, CudaScenario_id *cuda_device_scenario, unsigned int *temp_value) {
    for(size_t car_idx = GetGlobalIdx(); car_idx < cuda_device_scenario->getNumCars(); car_idx += GetGlobalDim()) {
        auto car = cuda_device_scenario->getCar(car_idx);
        size_t insert_offset = atomicAdd(temp_value + car->lane, 1);
        bucketmem->buckets[car->lane].buffer[insert_offset] = car;
        // printf("Insert: %lu at Lane(%lu) at %lu\n", car_idx, car->lane, insert_offset);
    }
}

CUDA_GLOB void bucketMemoryInitializeKernel(SortedBucketContainer *bucketmem,  BucketData *buckets, TrafficObject_id **main_buffer, CudaScenario_id *cuda_device_scenario, float bucket_memory_factor) {
    new(bucketmem)SortedBucketContainer(cuda_device_scenario, buckets, main_buffer, bucket_memory_factor);
}

CUDA_HOSTDEV size_t SortedBucketContainer::getBufferSize(CudaScenario_id &scenario, float bucket_memory_factor) {
    size_t total_buffer_size = 0;
    for (Lane_id &l : scenario.getLaneIterator()) {
        total_buffer_size += ceil(bucket_memory_factor * scenario.getRoad(l.road)->length / 5.);
    }
    return total_buffer_size;
}


CUDA_HOST size_t SortedBucketContainer::getBufferSize(Scenario_id &scenario, float bucket_memory_factor) {
    size_t total_buffer_size = 0;
    for (Lane_id &l : scenario.lanes) {
        total_buffer_size += ceil(bucket_memory_factor * scenario.roads.at(l.road).length / 5.);
    }
    return total_buffer_size;
}


CUDA_DEV SortedBucketContainer::SortedBucketContainer(CudaScenario_id *scenario, BucketData *_buckets, TrafficObject_id **_main_buffer, float bucket_memory_factor) {

    bucket_count = scenario->getNumLanes();

    buckets = _buckets; // new BucketData[scenario->getNumLanes()];
    main_buffer = _main_buffer; // new TrafficObject_id*[total_buffer_size];
    assert(main_buffer != nullptr);
    assert(buckets != nullptr);

    size_t total_buffer_size = 0;
    size_t i = 0;
    for (Lane_id &l : scenario->getLaneIterator()) {
        buckets[i].id = i;
        buckets[i].size = 0;
        buckets[i].buffer_size = ceil(bucket_memory_factor * scenario->getRoad(l.road)->length / 5.);
        buckets[i].buffer = main_buffer + total_buffer_size;
        total_buffer_size += buckets[i].buffer_size;
        i++;
    }

    this->main_buffer_size = total_buffer_size;
#ifdef DEBUG_MSGS
    printf("Allocated: %.2fMB\n", (float) (sizeof(size_t) * scenario->getNumLanes() * 2 + sizeof(TrafficObject_id**) * scenario->getNumLanes() +
                                           sizeof(TrafficObject_id*) * total_buffer_size) / 1024. / 1024.);
#endif
}

std::shared_ptr<SortedBucketContainer> SortedBucketContainer::fromScenario(Scenario_id &scenario, CudaScenario_id *device_cuda_scenario) {
    TrafficObject_id **main_buffer;
    gpuErrchk(cudaMalloc((void**) &main_buffer, SortedBucketContainer::getBufferSize(scenario, 4) * sizeof(TrafficObject_id*)));

    BucketData *buckets;
    gpuErrchk(cudaMalloc((void**) &buckets, scenario.lanes.size() * sizeof(BucketData)));

    // allocate bucket class
    SortedBucketContainer *bucket_memory;
    gpuErrchk(cudaMalloc((void**) &bucket_memory, sizeof(SortedBucketContainer)));

    // initialize bucket class
    bucketMemoryInitializeKernel<<<1, 1>>>(bucket_memory, buckets, main_buffer, device_cuda_scenario, 4.);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // load data into memory
    // bucketMemoryLoadKernel<<<1024, 1024>>>(bucket_memory, device_cuda_scenario, 4.);

    unsigned int *tmp;
    gpuErrchk(cudaMalloc((void**) &tmp, scenario.lanes.size() * sizeof(unsigned int )));
    gpuErrchk(cudaMemset(tmp, 0, scenario.lanes.size() * sizeof(unsigned int )));

    bucketMemoryLoadKernel2<<<1024, 1024>>>(bucket_memory, device_cuda_scenario, tmp);
    CHECK_FOR_ERROR();

    SortedBucketContainer::FixSize(bucket_memory, false);

    std::shared_ptr<SortedBucketContainer> result(bucket_memory, free_deleter());
    return result;
}



__device__ inline bool isInWrongLane(SortedBucketContainer *container, TrafficObject_id **object) {
    if (*object == nullptr) return false;
    BucketData supposed_bucket = container->buckets[(*object)->lane];
    return (object < supposed_bucket.buffer || supposed_bucket.buffer + supposed_bucket.size <= object);
}


__global__ void BlockWisePreScan(SortedBucketContainer *container, size_t *g_odata, size_t n) {
    assert(IsPowerOfTwo(n));
    extern __shared__ size_t temp[];// allocated on invocation

    size_t traffic_object_id = GetThreadIdx();
    size_t buffer_offset = GetBlockIdx() * n;
    assert(n == GetBlockDim() * 2);
    assert(GetGlobalDim() * 2 >= container->main_buffer_size);

    TrafficObject_id **p_obj1 = container->main_buffer + 2 * traffic_object_id + buffer_offset;
    TrafficObject_id **p_obj2 = container->main_buffer + 2 * traffic_object_id + 1 + buffer_offset;
    // if(GetGlobalIdx() == 0) printf("Starting...\n");

    if(p_obj1 < container->main_buffer + container->main_buffer_size && p_obj1 != nullptr) {
        temp[2 * traffic_object_id] = isInWrongLane(container, p_obj1) ? 1 : 0; // load input into shared memory
        //printf("Got one. %lu, %lu, %lu \n", traffic_object_id, buffer_offset, temp[2 * traffic_object_id]);
    } else
        temp[2 * traffic_object_id] = 0;

    if(p_obj2 < container->main_buffer + container->main_buffer_size && p_obj2 != nullptr) {
        temp[2 * traffic_object_id + 1] = isInWrongLane(container, p_obj2) ? 1 : 0;  // load input into shared memory
        //printf("Got one. %lu, %lu, %lu \n", traffic_object_id, buffer_offset, temp[2 * traffic_object_id + 1]);
    } else
        temp[2 * traffic_object_id + 1] = 0;

    PreScan(temp, traffic_object_id, n);

    __syncthreads();

    if(p_obj1 < container->main_buffer + container->main_buffer_size)
        g_odata[2 * traffic_object_id + buffer_offset] = temp[2*traffic_object_id]; // write results to device memory
    if(p_obj2 < container->main_buffer + container->main_buffer_size)
        g_odata[2 * traffic_object_id+1 + buffer_offset] = temp[2*traffic_object_id+1];

}

__global__ void MoveToReinsertBufferKernel(SortedBucketContainer *container, size_t *prefixSum, size_t n,
                                     TrafficObject_id **reinsert_buffer, size_t buffer_size) {

    size_t idx = GetGlobalIdx();

    assert(GetGlobalDim() >= n);

    if (idx >= n) return;

    if ((idx == 0 && prefixSum[0] > 0) || (idx != 0 && prefixSum[idx] != prefixSum[idx - 1])){
        size_t insert_id = idx == 0 ? 0 : prefixSum[idx - 1];
        if(insert_id >= buffer_size) {
            printf("%lu, %lu\n", insert_id, buffer_size);
        }

        if(!(insert_id < buffer_size && container->main_buffer[idx] != nullptr)) {
            printf("%lu: %lu - %lu\n", idx, prefixSum[idx], idx == 0 ? (size_t )-1 : prefixSum[idx - 1]);
        }

        assert(insert_id < buffer_size && container->main_buffer[idx] != nullptr);
        reinsert_buffer[insert_id] = container->main_buffer[idx];
        container->main_buffer[idx] = nullptr;
#ifdef RUN_WITH_TESTS
        if (reinsert_buffer[insert_id]->id == CAR_TO_ANALYZE) {
            printf("Car(%lu) moved to ReinsertBuffer(#%lu)\n", reinsert_buffer[insert_id]->id, insert_id);
        }
#endif
    }
}



__global__
void MoveToContainerKernel(SortedBucketContainer *container, TrafficObject_id **objectsToInsert, size_t *n, unsigned int *temp_value) {
    for(size_t idx = GetGlobalIdx(); idx < *n; idx += GetGlobalDim()) {

        int insert_offset = atomicAdd(temp_value + objectsToInsert[idx]->lane, 1);
        auto &bucket = container->buckets[objectsToInsert[idx]->lane];
        assert(bucket.size + insert_offset < bucket.buffer_size);
#ifdef DEBUG_MSGS
        if(bucket.size + insert_offset >= bucket.buffer_size) {
            printf("Buffer-Overflow for Lane(%lu)\n", objectsToInsert[idx]->lane);
        }
#endif
        bucket.buffer[bucket.size + insert_offset] = objectsToInsert[idx];
#ifdef RUN_WITH_TESTS
        if (objectsToInsert[idx]->id == CAR_TO_ANALYZE) {
            printf("Car(%lu) from ReinsertBuffer(#%lu) to Bucket(%lu)\n", objectsToInsert[idx]->id, idx, objectsToInsert[idx]->lane);
        }
#endif
    }
}

void MyPreScan(Scenario_id &scenario, SortedBucketContainer *container, SortBuffer &sortBuffer) {

    assert(IsPowerOfTwo(PRE_SUM_BLOCK_SIZE));

    size_t buffer_size = SortedBucketContainer::getBufferSize(scenario, 4.);

    BlockWisePreScan<<<buffer_size / PRE_SUM_BLOCK_SIZE + 1, PRE_SUM_BLOCK_SIZE / 2, PRE_SUM_BLOCK_SIZE * sizeof(size_t)>>>(container, sortBuffer.preSumOut, PRE_SUM_BLOCK_SIZE);
    CHECK_FOR_ERROR();

#ifdef RUN_WITH_TESTS
    std::vector<size_t> preSumHost(buffer_size);
    gpuErrchk(cudaMemcpy(preSumHost.data(), sortBuffer.temporary_pre_sum_buffers[0], sortBuffer.temporary_pre_sum_buffer_sizes[0] * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_FOR_ERROR();
#endif

    size_t *previousPreSumTemp, *preSumTemp;
    size_t previousPreSumTempSize, preSumTempSize;
    for(size_t i=1; i < sortBuffer.temporary_pre_sum_buffers.size(); i++) {
        preSumTemp = sortBuffer.temporary_pre_sum_buffers[i];
        preSumTempSize = sortBuffer.temporary_pre_sum_buffer_sizes[i];
        previousPreSumTemp = sortBuffer.temporary_pre_sum_buffers[i - 1];
        previousPreSumTempSize = sortBuffer.temporary_pre_sum_buffer_sizes[i - 1];

        MergeBlockWisePreScan<<<preSumTempSize / PRE_SUM_BLOCK_SIZE + 1, PRE_SUM_BLOCK_SIZE / 2, PRE_SUM_BLOCK_SIZE * sizeof(size_t )>>>(preSumTemp, preSumTempSize, previousPreSumTemp, previousPreSumTempSize, PRE_SUM_BLOCK_SIZE, PRE_SUM_BLOCK_SIZE);
        CHECK_FOR_ERROR();

#ifdef RUN_WITH_TESTS
        gpuErrchk(cudaMemcpy(preSumHost.data(), preSumTemp, preSumTempSize * sizeof(size_t), cudaMemcpyDeviceToHost));
        CHECK_FOR_ERROR();
#endif
    }

    for(size_t i=sortBuffer.temporary_pre_sum_buffer_sizes.size() - 1; i > 0; i--) {
        preSumTemp = sortBuffer.temporary_pre_sum_buffers[i];
        preSumTempSize = sortBuffer.temporary_pre_sum_buffer_sizes[i];
        previousPreSumTemp = sortBuffer.temporary_pre_sum_buffers[i - 1];
        previousPreSumTempSize = sortBuffer.temporary_pre_sum_buffer_sizes[i - 1];

#ifdef RUN_WITH_TESTS
        gpuErrchk(cudaMemcpy(preSumHost.data(), preSumTemp, preSumTempSize * sizeof(size_t), cudaMemcpyDeviceToHost));
        CHECK_FOR_ERROR();
#endif

        MergeBlockWisePreScanStep2<<<preSumTempSize, PRE_SUM_BLOCK_SIZE, PRE_SUM_BLOCK_SIZE * sizeof(size_t )>>>(previousPreSumTemp, preSumTemp, PRE_SUM_BLOCK_SIZE, previousPreSumTempSize, previousPreSumTempSize);
        CHECK_FOR_ERROR();
    }
#ifdef RUN_WITH_TESTS
    gpuErrchk(cudaMemcpy(preSumHost.data(), sortBuffer.temporary_pre_sum_buffers[0], sortBuffer.temporary_pre_sum_buffer_sizes[0] * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_FOR_ERROR();
#endif

}


__global__ void GetIsInWrongLaneKernel(SortedBucketContainer *container, size_t car_count, size_t *sizes, size_t num_sizes,
        size_t *lanePreSum, size_t lanePreSumSize) {
    assert(num_sizes >= container->main_buffer_size);
    for(size_t i=GetGlobalIdx(); i < car_count; i += GetGlobalDim()) {
        size_t element_idx, bucket_idx;
        GetBucketIdxFromGlobalIdx(i, lanePreSum, lanePreSumSize, &bucket_idx, &element_idx);
        if (bucket_idx >= container->bucket_count) return;
#ifdef DEBUG_MSGS
        if (element_idx >= container->buckets[bucket_idx].size)
            printf("%lu: %lu, %lu, %lu, %lu\n", i,  bucket_idx, container->bucket_count, element_idx, container->buckets[bucket_idx].size);
#endif
        assert(bucket_idx < container->bucket_count);
        assert(element_idx < container->buckets[bucket_idx].size);
        sizes[i] = (size_t) (int) isInWrongLane(container, container->buckets[bucket_idx].buffer + element_idx);
        // printf("%lu - %lu.%lu: Lane(%lu) : %lu\n", i, bucket_idx, element_idx, container->buckets[bucket_idx].buffer[element_idx]->lane, sizes[i]);
    }
}



__global__ void MoveToReinsertBufferKernel2(SortedBucketContainer *container, size_t *prefixSum, size_t n,
                                           TrafficObject_id **reinsert_buffer, size_t buffer_size,
                                            size_t *lanePreSum, size_t lanePreSumSize) {
    for(size_t idx = GetGlobalIdx(); idx < n; idx += GetGlobalDim()) {
        size_t element_idx, bucket_idx;
        GetBucketIdxFromGlobalIdx(idx, lanePreSum, lanePreSumSize, &bucket_idx, &element_idx);
        if (bucket_idx >= container->bucket_count) return;
        if (element_idx >= container->buckets[bucket_idx].size)
            printf("%lu: %lu, %lu, %lu, %lu\n", idx,  bucket_idx, container->bucket_count, element_idx, container->buckets[bucket_idx].size);

        assert(bucket_idx < container->bucket_count);
        assert(element_idx < container->buckets[bucket_idx].size);

        if ((idx == 0 && prefixSum[0] > 0) || (idx != 0 && prefixSum[idx] != prefixSum[idx - 1])){
            size_t insert_id = idx == 0 ? 0 : prefixSum[idx - 1];
            assert(insert_id < buffer_size && container->buckets[bucket_idx].buffer[element_idx] != nullptr);
            reinsert_buffer[insert_id] = container->buckets[bucket_idx].buffer[element_idx];
            container->buckets[bucket_idx].buffer[element_idx] = nullptr;

#ifdef RUN_WITH_TESTS
            //if (reinsert_buffer[insert_id]->id == CAR_TO_ANALYZE)
                printf("Car(%lu, %lu -> %lu) moved to ReinsertBuffer(#%lu)\n", reinsert_buffer[insert_id]->id, bucket_idx, reinsert_buffer[insert_id]->lane, insert_id);
#endif
        }
    }
}


int i = 0;
void RestoreCorrectBucket(Scenario_id &scenario, SortedBucketContainer *container, SortBuffer &sortBuffer) {

    size_t number_of_lanes = scenario.lanes.size();
    size_t number_of_cars = scenario.cars.size();

    size_t buffer_size = SortedBucketContainer::getBufferSize(scenario, 4.);

    SortedBucketContainer::FetchBucketSizes(container, scenario, sortBuffer.bucketSizes);

    CalculatePreSum(sortBuffer.laneBucketPreSumBuffer, sortBuffer.lanePreSumBufferSize, sortBuffer.bucketSizes, scenario.lanes.size(), sortBuffer.preSumBatchSize);
    CHECK_FOR_ERROR();

    GetIsInWrongLaneKernel<<<SUGGESTED_THREADS, SUGGESTED_THREADS>>>
        (container, scenario.cars.size(), sortBuffer.preSumIn, sortBuffer.preSumInLen, sortBuffer.laneBucketPreSumBuffer, scenario.lanes.size());
    CHECK_FOR_ERROR();
    CalculatePreSum(sortBuffer.preSumOut, sortBuffer.preSumOutLen, sortBuffer.preSumIn, scenario.cars.size(), sortBuffer.batch_count);


    MoveToReinsertBufferKernel2<<<buffer_size / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>
        (container, sortBuffer.preSumOut, scenario.cars.size(), sortBuffer.reinsert_buffer, sortBuffer.reinsert_buffer_size,
                sortBuffer.laneBucketPreSumBuffer, scenario.lanes.size());
    CHECK_FOR_ERROR();
    i++;

#ifdef RUN_WITH_TESTS
    std::vector<size_t> preSumHost(buffer_size);
    std::vector<size_t> preSumHostO(buffer_size);
    gpuErrchk(cudaMemcpy(preSumHostO.data(), sortBuffer.preSumOut, scenario.cars.size() * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_FOR_ERROR();
    gpuErrchk(cudaMemcpy(preSumHost.data(), sortBuffer.preSumIn, scenario.cars.size() * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_FOR_ERROR();
#endif

    /* MyPreScan(scenario, container, sortBuffer);

    MoveToReinsertBufferKernel<<<buffer_size / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>
        (container, sortBuffer.preSumOut, buffer_size, sortBuffer.reinsert_buffer, sortBuffer.reinsert_buffer_size);
    CHECK_FOR_ERROR()*/

    gpuErrchk(cudaMemsetAsync(sortBuffer.multiScanTmp, 0, sortBuffer.multiScanTmpBytes));
    MoveToContainerKernel<<<256, 256>>>(container, sortBuffer.reinsert_buffer, sortBuffer.preSumOut + scenario.cars.size() - 1, sortBuffer.multiScanTmp);
    CHECK_FOR_ERROR();

    SetTempSizeKernel<<<number_of_lanes / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>(container, sortBuffer.multiScanTmp);
    CHECK_FOR_ERROR();

}

void SortedBucketContainer::RestoreValidState(Scenario_id &scenario, SortedBucketContainer *container, SortBuffer &sortBuffer) {

    RestoreCorrectBucket(scenario, container, sortBuffer);

    Sort(container, scenario, sortBuffer);

    SortedBucketContainer::FixSize(container, scenario, true, sortBuffer);

}