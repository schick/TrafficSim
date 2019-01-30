//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_OPENMPALGORITHM_H
#define PROJECT_OPENMPALGORITHM_H

#include "RefAlgorithm.h"
#include "model/Scenario.h"
#include "optimization/model/OptimizeScenario.h"

#ifdef VISUALIZATION_ENABLED
#include "Visualization.h"
#endif

class OpenMPAlgorithm : public RefAlgorithm {

public:
    ADVANCE_ALGO_INIT_WITH_OPT(OpenMPAlgorithm, Scenario, OptimizeScenario, Visualization);
    
    explicit OpenMPAlgorithm(std::shared_ptr<BaseScenario> scenario) : RefAlgorithm(scenario) {};

    void prepareCars();

    void advanceCars();
    void advanceTrafficLights();
    void advance(size_t steps) override;
    void sortLanes();
};

#endif //PROJECT_OPENMPALGORITHM_H
