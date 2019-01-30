//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_SEQUENTIALALGORITHM_H
#define PROJECT_SEQUENTIALALGORITHM_H

#include "AdvanceAlgorithm.h"
#include "model/Scenario.h"
#include "optimization/model/OptimizeScenario.h"
#include "RefAlgorithm.h"

#ifdef VISUALIZATION_ENABLED
#include "Visualization.h"
#endif

class SequentialAlgorithm : public RefAlgorithm {

public:
    ADVANCE_ALGO_INIT_WITH_OPT(SequentialAlgorithm, Scenario, OptimizeScenario, Visualization);

    explicit SequentialAlgorithm(std::shared_ptr<BaseScenario> scenario) : RefAlgorithm(scenario) {};

    void prepareCars();

    void advanceCars();
    void advanceTrafficLights();
    void advance(std::size_t steps) override;
    void cacheNeighbors();
    void sortLanes();
};

#endif //PROJECT_SEQUENTIALALGORITHM_H
