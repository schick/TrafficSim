//
// Created by oke on 17.01.19.
//

#include "SortBuffer.h"
#include "cuda/cuda_utils.h"
#include "SortedBucketContainer.h"
#include "PreScan.h"

#define MAX(a, b) (a < b ? b : a)
#define MIN(a, b) (a < b ? a : b)

SortBuffer::SortBuffer(Scenario_id &scenario, size_t preSumBatchSize) {

    this->preSumBatchSize = preSumBatchSize;

    GPU_ALLOC((void **) &bucketSizes, sizeof(size_t) * scenario.lanes.size())


    lanePreSumBufferSize = GetRequiredPreSumReqBufferSize(scenario.lanes.size(), preSumBatchSize);
    GPU_ALLOC((void **) &laneBucketPreSumBuffer, sizeof(size_t) * lanePreSumBufferSize)

    GPU_ALLOC((void **) &pBucketData, scenario.lanes.size() * sizeof(BucketData))
    GPU_ALLOC((void **) &pBucketData2, scenario.lanes.size() * sizeof(BucketData))
    GPU_ALLOC((void **) &pBucketDataNumFilled, sizeof(unsigned int))
    GPU_ALLOC((void **) &pBucketDataNumFilled2, sizeof(unsigned int))

    laneCounterSize = scenario.lanes.size();
    GPU_ALLOC((void **) &laneCounter, laneCounterSize * sizeof(unsigned int))

    reinsert_buffer_size = scenario.cars.size();
    GPU_ALLOC((void **) &reinsert_buffer, reinsert_buffer_size * sizeof(TrafficObject_id *))

    preSumInLen = MAX(scenario.cars.size(), scenario.lanes.size());
    preSumOutLen = GetRequiredPreSumReqBufferSize(preSumInLen, batch_count);
    assert(IsPowerOfTwo(PRE_SUM_BLOCK_SIZE));

    GPU_ALLOC((void **) &preSumIn, preSumInLen * sizeof(size_t))
    GPU_ALLOC((void **) &preSumOut, preSumOutLen * sizeof(size_t))
}

SortBuffer::~SortBuffer() {

    cudaFree(preSumIn);
    cudaFree(preSumOut);

    cudaFree(reinsert_buffer);

    cudaFree(laneCounter);

    cudaFree(pBucketDataNumFilled);
    cudaFree(pBucketDataNumFilled2);
    cudaFree(pBucketData);
    cudaFree(pBucketData2);

    cudaFree(laneBucketPreSumBuffer);

    cudaFree(bucketSizes);

}