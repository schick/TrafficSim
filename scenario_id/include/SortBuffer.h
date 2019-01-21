//
// Created by oke on 17.01.19.
//

#ifndef TRAFFIC_SIM_SORTBUFFER_H
#define TRAFFIC_SIM_SORTBUFFER_H

#include <vector>

#include "Scenario_id.h"
#include "cuda/cuda_utils.h"
class BucketData;

class SortBuffer {
public:
    size_t batch_count = SUGGESTED_THREADS;
    size_t *bucketSizes, *bucketSizePreSum;

    size_t *preSumIn;
    size_t preSumInLen;
    size_t *preSumOut;
    size_t preSumOutLen;

    unsigned int *laneCounter;
    size_t laneCounterSize;

    TrafficObject_id **reinsert_buffer;
    size_t reinsert_buffer_size;
    size_t multiScanTmpBytes;

    BucketData *pBucketData;
    BucketData *pBucketData2;

    unsigned int *pBucketDataNumFilled;
    unsigned int *pBucketDataNumFilled2;

    size_t preSumBatchSize;
    size_t lanePreSumBufferSize;
    size_t *laneBucketPreSumBuffer;


    SortBuffer(Scenario_id &scenario, size_t preSumBatchSize);
    ~SortBuffer();
};



#endif //TRAFFIC_SIM_SORTBUFFER_H
