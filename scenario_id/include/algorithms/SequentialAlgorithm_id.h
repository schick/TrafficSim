//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_SEQUENTIALALGORITHM_ID_H
#define PROJECT_SEQUENTIALALGORITHM_ID_H

#include "AdvanceAlgorithm.h"
#include "Car_id.h"
#include "Scenario_id.h"
#include "Visualization_id.h"


class SequentialAlgorithm_id : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(SequentialAlgorithm_id, Scenario_id, Visualization_id);

    explicit SequentialAlgorithm_id(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario) {};

    std::vector<Car_id::AdvanceData> calculateCarChanges();

    Scenario_id* getIDScenario() {
        return dynamic_cast<Scenario_id*>(getScenario().get());
    }


    void advanceCars();
    void advanceTrafficLights();
    void advance(size_t steps) override;

};

#endif //PROJECT_SEQUENTIALALGORITHM_H
