//
// Created by maxi on 1/27/19.
//

#include "optimization/SequentialRandomOptimizer.h"
#include "AdvanceAlgorithm.h"
#include "optimization/model/OptimizeScenario.h"
#include "optimization/model/SignalLayout.h"

#include <memory>
#include <stdexcept>

nlohmann::json SequentialRandomOptimizer::optimize() {

    while (true) {

        iterations += 1;

        SignalLayout sigLayout(algorithm, scenarioData);

        if (sigLayout.getTravelledDistance() > minTravelLength) {
            return sigLayout.toJson();
        }
    }
}
