//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_TESTALGO_H
#define PROJECT_TESTALGO_H

#include "algorithm"
#include "AdvanceAlgorithm.h"
#include "model/Car_id.h"
#include "model/Scenario_id.h"
#include "Visualization_id.h"
#include <list>
#include <map>
#include "AlgorithmWrapper.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif

class CudaAlgorithm : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT_WITH_OPT(CudaAlgorithm, Scenario_id, Scenario_id, Visualization_id);

    explicit CudaAlgorithm(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};

    Scenario_id* getIDScenario() {
        return dynamic_cast<Scenario_id*>(getScenario().get());
    }

    void advance(size_t steps) override;
};

#endif //PROJECT_SEQUENTIALALGORITHM_H
