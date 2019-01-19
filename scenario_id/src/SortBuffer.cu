//
// Created by oke on 17.01.19.
//

#include "SortBuffer.h"
#include "cuda/cuda_utils.h"
#include "SortedBucketContainer.h"
#include "PreScan.h"

SortBuffer::SortBuffer(Scenario_id &scenario, size_t preSumBatchSize) {

    this->preSumBatchSize = preSumBatchSize;

    gpuErrchk(cudaMalloc((void **) &laneBucketSizeBuffer, sizeof(size_t) * scenario.lanes.size()));
    gpuErrchk(cudaMalloc((void **) &bucketSizes, sizeof(size_t) * scenario.lanes.size()));
    lanePreSumBufferSize = GetRequiredPreSumReqBufferSize(scenario.lanes.size(), preSumBatchSize);
    gpuErrchk(cudaMalloc((void **) &laneBucketPreSumBuffer, sizeof(size_t) * lanePreSumBufferSize));

    gpuErrchk(cudaMalloc((void **) &pBucketData, scenario.lanes.size() * sizeof(BucketData)));
    gpuErrchk(cudaMalloc((void **) &pBucketData2, scenario.lanes.size() * sizeof(BucketData)));

    gpuErrchk(cudaMalloc((void **) &last, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **) &last2, sizeof(unsigned int)));


    multiScanTmpBytes = 2 * scenario.lanes.size() * sizeof(unsigned int);
    gpuErrchk(cudaMalloc((void **) &multiScanTmp, multiScanTmpBytes * sizeof(unsigned int)));

    reinsert_buffer_size = scenario.lanes.size();
    gpuErrchk(cudaMalloc((void **) &reinsert_buffer, reinsert_buffer_size * sizeof(TrafficObject_id *)));

    size_t buffer_size = SortedBucketContainer::getBufferSize(scenario, 4.);
    preSumInLen = buffer_size;
    preSumOutLen = GetRequiredPreSumReqBufferSize(buffer_size, batch_count);
    assert(IsPowerOfTwo(PRE_SUM_BLOCK_SIZE));

    gpuErrchk(cudaMalloc((void **) &preSumIn, preSumInLen * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void **) &preSumOut, preSumOutLen * sizeof(size_t)));

    temporary_pre_sum_buffers.push_back(preSumOut);
    temporary_pre_sum_buffer_sizes.push_back(buffer_size);


    size_t preSumTempSize = buffer_size;
    size_t *preSumTemp;
    for (size_t i = PRE_SUM_BLOCK_SIZE; i < buffer_size; i *= PRE_SUM_BLOCK_SIZE) {
        preSumTempSize = preSumTempSize / PRE_SUM_BLOCK_SIZE + 1;
        gpuErrchk(cudaMalloc((void **) &preSumTemp, preSumTempSize * sizeof(size_t)));
        temporary_pre_sum_buffers.push_back(preSumTemp);
        temporary_pre_sum_buffer_sizes.push_back(preSumTempSize);
    }
}

SortBuffer::~SortBuffer() {
    for(size_t * buffer : temporary_pre_sum_buffers) {
        cudaFree(buffer);
    }

    cudaFree(multiScanTmp);
    cudaFree(reinsert_buffer);
    cudaFree(last);
    cudaFree(last2);
    cudaFree(pBucketData);
    cudaFree(pBucketData2);

}