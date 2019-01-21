//
// Created by oke on 17.01.19.
//

#include "SortBuffer.h"
#include "cuda/cuda_utils.h"
#include "SortedBucketContainer.h"
#include "PreScan.h"

SortBuffer::SortBuffer(Scenario_id &scenario, size_t preSumBatchSize) {

    this->preSumBatchSize = preSumBatchSize;

    gpuErrchk(cudaMalloc((void **) &bucketSizes, sizeof(size_t) * scenario.lanes.size()));

    lanePreSumBufferSize = GetRequiredPreSumReqBufferSize(scenario.lanes.size(), preSumBatchSize);
    gpuErrchk(cudaMalloc((void **) &laneBucketPreSumBuffer, sizeof(size_t) * lanePreSumBufferSize));

    gpuErrchk(cudaMalloc((void **) &pBucketData, scenario.lanes.size() * sizeof(BucketData)));
    gpuErrchk(cudaMalloc((void **) &pBucketData2, scenario.lanes.size() * sizeof(BucketData)));
    gpuErrchk(cudaMalloc((void **) &pBucketDataNumFilled, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **) &pBucketDataNumFilled2, sizeof(unsigned int)));

    laneCounterSize = scenario.lanes.size();
    gpuErrchk(cudaMalloc((void **) &laneCounter, laneCounterSize * sizeof(unsigned int)));

    reinsert_buffer_size = scenario.lanes.size();
    gpuErrchk(cudaMalloc((void **) &reinsert_buffer, reinsert_buffer_size * sizeof(TrafficObject_id *)));

    size_t buffer_size = SortedBucketContainer::getBufferSize(scenario, 4.);
    preSumInLen = buffer_size;
    preSumOutLen = GetRequiredPreSumReqBufferSize(buffer_size, batch_count);
    assert(IsPowerOfTwo(PRE_SUM_BLOCK_SIZE));

    gpuErrchk(cudaMalloc((void **) &preSumIn, preSumInLen * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void **) &preSumOut, preSumOutLen * sizeof(size_t)));

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