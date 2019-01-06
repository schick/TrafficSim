//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_OPENMPALGORITHM_H
#define PROJECT_OPENMPALGORITHM_H

#include "AdvanceAlgorithm.h"
#include "model/Scenario.h"
#include "Visualization.h"
#include "IntelligentDriverModel.h"

class OpenMPAlgorithm : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(OpenMPAlgorithm, Scenario, Visualization);


    explicit OpenMPAlgorithm( std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};

    void calculateCarChanges();

    Scenario* getRefScenario() {
        return dynamic_cast<Scenario*>(getScenario().get());
    }
    void advanceCars();
    void advanceTrafficLights();
    void advance(size_t steps) override;
    void sortLanes();
};

#endif //PROJECT_OPENMPALGORITHM_H
