//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_OPENMPALGORITHM_H
#define PROJECT_OPENMPALGORITHM_H

#include "AdvanceAlgorithm.h"
#include "Scenario.h"
#include "Visualization.h"

class OpenMPAlgorithm : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(OpenMPAlgorithm, Scenario, Visualization);


    explicit OpenMPAlgorithm( std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};

    Scenario* getRefScenario() {
        return dynamic_cast<Scenario*>(getScenario().get());
    }

    void advance(size_t steps) override;
};

#endif //PROJECT_OPENMPALGORITHM_H
