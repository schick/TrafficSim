//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_SEQUENTIALALGORITHM_H
#define PROJECT_SEQUENTIALALGORITHM_H

#include "AdvanceAlgorithm.h"
#include "model/Scenario.h"
#include "IntelligentDriverModel.h"
#ifdef VISUALIZATION_ENABLED
#include "Visualization.h"
#endif

class SequentialAlgorithm : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(SequentialAlgorithm, Scenario, Visualization);

    explicit SequentialAlgorithm(std::shared_ptr<BaseScenario>scenario) : AdvanceAlgorithm(scenario) {};

    void calculateCarChanges();

    Scenario* getRefScenario() {
        return dynamic_cast<Scenario*>(getScenario().get());
    }
    void advanceCars();
    void advanceTrafficLights();
    void advance(std::size_t steps) override;
    void sortLanes();
};

#endif //PROJECT_SEQUENTIALALGORITHM_H
