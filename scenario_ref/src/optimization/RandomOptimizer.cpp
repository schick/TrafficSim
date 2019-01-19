//
// Created by maxi on 1/18/19.
//

#include "optimization/RandomOptimizer.h"
#include "optimization/model/OptimizeScenario.h"
#include "AdvanceAlgorithm.h"
#include <memory>

#include <random>

inline unsigned int range_random(size_t min, size_t max) {
    return rand() % (max - min) + min;
}

inline unsigned int range_random(size_t max) {
    return range_random(0, max);
}


void randomInitialization(OptimizeScenario &scenario) {
    for(auto &junction : scenario.junctions) {
        junction.signals.resize(range_random(2, 5));
        for(auto &signal : junction.signals) {
            do {
                signal.direction = (Junction::Direction ) range_random(4);
            } while(junction.incoming[signal.direction] == nullptr);
            signal.duration = range_random(5, 11);
        }
    }
    scenario.initJunctions();
}

void RandomOptimizer::optimize() {

    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
    if (advancer == nullptr) {
        printf("Algorithm not found.");
        exit(-1);
    }

    OptimizeScenario &scenario = *dynamic_cast<OptimizeScenario *>(advancer->getScenario().get());

    randomInitialization(scenario);

    advancer->advance(scenarioData["time_steps"]);

    double total_distance = scenario.getTraveledDistance();

    printf("Distance: %.2f", total_distance);

}
