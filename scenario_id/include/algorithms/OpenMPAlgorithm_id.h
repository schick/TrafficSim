//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_OPENMPALGORITHM_ID_H
#define PROJECT_OPENMPALGORITHM_ID_H

#include "AdvanceAlgorithm.h"
#include "Scenario_id.h"
#include "Visualization_id.h"

class OpenMPAlgorithm_id : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(OpenMPAlgorithm_id, Scenario_id, Visualization_id);

    explicit OpenMPAlgorithm_id(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};


    Scenario_id* getIDScenario() {
        return dynamic_cast<Scenario_id*>(getScenario().get());
    }

    void advance(size_t steps) override;
};

#endif //PROJECT_OPENMPALGORITHM_H
