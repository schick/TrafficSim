//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_OPENMPALGORITHM_H
#define PROJECT_OPENMPALGORITHM_H

#include "AdvanceAlgorithm.h"
#include "model/Scenario.h"
#include "optimization/model/OptimizeScenario.h"

#ifdef VISUALIZATION_ENABLED
#include "Visualization.h"
#endif

class OpenMPAlgorithm : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT_WITH_OPT(OpenMPAlgorithm, Scenario, OptimizeScenario, Visualization);
    
    explicit OpenMPAlgorithm(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};

    void prepareCars();

    Scenario* getRefScenario() {
        return dynamic_cast<Scenario*>(getScenario().get());
    }
    void advanceCars();
    void advanceTrafficLights();
    void advance(size_t steps) override;
    void sortLanes();
};

#endif //PROJECT_OPENMPALGORITHM_H
