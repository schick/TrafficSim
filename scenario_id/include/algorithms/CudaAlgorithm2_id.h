//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_CUDAALGORITHM2_ID_H
#define PROJECT_CUDAALGORITHM2_ID_H

#include <memory>

#include "AdvanceAlgorithm.h"
#include "Car_id.h"
#include "Scenario_id.h"
#include "Visualization_id.h"
#include "cudacontainer.h"
#include "AlgorithmWrapper.h"


class TrafficObjectBucket {

    TrafficObjectBucket() : objects(nullptr), size(0) {}
    TrafficObjectBucket(size_t size) {
        this->size = size;
        objects = (TrafficObject_id*) malloc(size * sizeof(TrafficObject_id));
    }

    // move constructor
    TrafficObjectBucket(TrafficObjectBucket&& other) {
        other.size = 0;
        other.objects = nullptr;
    }

    // copy constructor
    TrafficObjectBucket(TrafficObjectBucket& other) {
        if (size != other.size && objects != nullptr) {
            free(objects);
            size = other.size;
            objects = (TrafficObject_id*) malloc(size * sizeof(TrafficObject_id));
        }
        for(int i=0; i < size; i++) objects[i] = other.objects[i];
    }
    // destructor
    ~TrafficObjectBucket() {
        free(objects);
    }

    size_t size;
    TrafficObject_id *objects;
};


class TrafficObjectBucketContainer {

public:

private:


};



class CudaAlgorithm2_id : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(CudaAlgorithm2_id, Scenario_id, Visualization_id);

    explicit CudaAlgorithm2_id(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario),
            device_cuda_scenario(CudaScenario_id::fromScenarioData_device(*getIDScenario())) {






    }

    Scenario_id* getIDScenario() {
        return dynamic_cast<Scenario_id*>(getScenario().get());
    }

    void advance(size_t steps) override;

private:

    CudaScenario_id *device_cuda_scenario;


};

#endif //PROJECT_SEQUENTIALALGORITHM_H
