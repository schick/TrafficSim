//
// Created by maxi on 1/18/19.
//

#include "optimization/RandomOptimizer.h"
#include "optimization/model/OptimizeScenario.h"
#include "AdvanceAlgorithm.h"
#include <memory>
#include <stdexcept>
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

nlohmann::json RandomOptimizer::optimize() {

    while (true) {

        std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
        if (advancer == nullptr) {
            throw std::runtime_error("Algorithm not found: " + algorithm);
        }

        OptimizeScenario *scenario = dynamic_cast<OptimizeScenario *>(advancer->getScenario().get());
        if (scenario == nullptr) {
            throw std::runtime_error("Algorithm '" + algorithm + "' with wrong scenario type for 'RandomOptimizer'");
        }
        randomInitialization(*scenario);
        advancer->advance(scenarioData["time_steps"]);

        double total_distance = scenario->getTravelledDistance();

#ifdef DEBUG_MSGS
        printf("Distance: %.2f\n", total_distance);
#endif

        if (total_distance > minTravelLength) {
            return scenario->toJson();
        }
    }


}
