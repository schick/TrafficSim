//
// Created by oke on 17.01.19.
//

#ifndef TRAFFIC_SIM_BUCKETMEMORY_H
#define TRAFFIC_SIM_BUCKETMEMORY_H

#include "cuda_utils/cuda_utils.h"
#include "model/TrafficObject_id.h"
#include "algorithms/AlgorithmWrapper.h"
#include "SortBuffer.h"

#define BUCKET_TO_ANALYZE 1110
#define CAR_TO_ANALYZE 133 // 308


struct BucketData {
    size_t id;
    size_t size;
    size_t buffer_size;
    TrafficObject_id **buffer;
};

class SortedBucketContainer {
private:
public:
    size_t bucket_count;
    BucketData *buckets;
    TrafficObject_id **main_buffer;
    size_t main_buffer_size;


    CUDA_HOSTDEV SortedBucketContainer() : bucket_count(0), buckets(nullptr), main_buffer(nullptr), main_buffer_size(0) {}
    CUDA_HOSTDEV SortedBucketContainer(CudaScenario_id *scenario, BucketData *_buckets, TrafficObject_id **_main_buffer, float bucket_memory_factor);


    // CUDA_HOST static void FixSize(SortedBucketContainer *bucketMemory, bool only_lower);
    CUDA_HOST static void FixSize(SortedBucketContainer *bucketMemory, Scenario_id &scenario, bool only_lower, SortBuffer &sortBuffer);

    CUDA_HOSTDEV static size_t getBufferSize(CudaScenario_id &scenario, float bucket_memory_factor);
    CUDA_HOST static size_t getBufferSize(Scenario_id &scenario, float bucket_memory_factor);

    CUDA_HOST static std::shared_ptr<SortedBucketContainer> fromScenario(Scenario_id &scenario, CudaScenario_id *device_cuda_scenario, SortBuffer &sortBuffer);

    CUDA_HOST static void Sort(SortedBucketContainer *container, Scenario_id &scenario, SortBuffer &sortBuffer);
    CUDA_HOST static void FetchBucketSizes(SortedBucketContainer *container, Scenario_id &scenario, size_t *bucketSizes);
    CUDA_HOST static void FetchBucketBufferSizes(SortedBucketContainer *container, Scenario_id &scenario, size_t *bucketSizes);
    CUDA_HOST static void RestoreValidState(Scenario_id &scenario, SortedBucketContainer *container, SortBuffer &sortBuffer);



    static void FetchBucketSizes(BucketData *buckets, size_t num_buckets, Scenario_id &scenario, size_t *bucketSizes);
    static void SortLargeBuckets(BucketData *buckets, size_t num_buckets, Scenario_id &scenario, SortBuffer &sortBuffer);
    static void SortInSizeSteps(SortedBucketContainer *container, Scenario_id &scenario, SortBuffer &sortBuffer);
    static void SortFixedSize(SortedBucketContainer *container, Scenario_id &scenario, SortBuffer &sortBuffer);
};



#endif //TRAFFIC_SIM_BUCKETMEMORY_H
