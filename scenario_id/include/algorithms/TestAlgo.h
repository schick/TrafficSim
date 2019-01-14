//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_TESTALGO_H
#define PROJECT_TESTALGO_H

#include "algorithm"
#include "AdvanceAlgorithm.h"
#include "Car_id.h"
#include "Scenario_id.h"
#include "Visualization_id.h"
#include <list>
#include <map>
#include "AlgorithmWrapper.h"


#define MAX(a, b) (a < b ? b : a)
#define MIN(a, b) (a < b ? a : b)

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif



struct BucketData {
    size_t size;
    size_t buffer_size;
    TrafficObject_id **buffer;
};


class BucketMemory {
private:
public:
    size_t bucket_count;
    BucketData *buckets;
    TrafficObject_id **main_buffer;
    size_t main_buffer_size;


    CUDA_HOSTDEV BucketMemory() : bucket_count(0), buckets(nullptr), main_buffer(nullptr), main_buffer_size(0) {}
    CUDA_HOSTDEV BucketMemory(CudaScenario_id *scenario, BucketData *_buckets, TrafficObject_id **_main_buffer, float bucket_memory_factor);

    CUDA_HOSTDEV static size_t getBufferSize(CudaScenario_id &scenario, float bucket_memory_factor);
    CUDA_HOST static size_t getBufferSize(Scenario_id &scenario, float bucket_memory_factor);

    CUDA_HOST static std::shared_ptr<BucketMemory> fromScenario(Scenario_id &scenario, CudaScenario_id *device_cuda_scenario);

    CUDA_HOST static void test(BucketMemory *memory);
};

class TestAlgo : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(TestAlgo, Scenario_id, Visualization_id);

    explicit TestAlgo(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};

    Scenario_id* getIDScenario() {
        return dynamic_cast<Scenario_id*>(getScenario().get());
    }

    void advance(size_t steps) override;
};

#endif //PROJECT_SEQUENTIALALGORITHM_H
