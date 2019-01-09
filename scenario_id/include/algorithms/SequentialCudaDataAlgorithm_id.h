//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_SEQUENTIALALGORITHM_ID_H
#define PROJECT_SEQUENTIALALGORITHM_ID_H

#include "AdvanceAlgorithm.h"
#include "Car_id.h"
#include "Scenario_id.h"
#include "Visualization_id.h"
#include "AlgorithmWrapper.h"


class SequentialCudaDataAlgorithm_id : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(SequentialCudaDataAlgorithm_id, Scenario_id, Visualization_id);

    explicit SequentialCudaDataAlgorithm_id(std::shared_ptr<BaseScenario> scenario)
        : AdvanceAlgorithm(scenario),
        cudaScenario(CudaScenario_id::fromScenarioData(*dynamic_cast<Scenario_id*>(scenario.get()))),
        wrapper(cudaScenario) {
    };

    void advance(size_t steps) override;

private:
    CudaScenario_id cudaScenario;
    AlgorithmWrapper wrapper;
};

#endif //PROJECT_SEQUENTIALALGORITHM_H
