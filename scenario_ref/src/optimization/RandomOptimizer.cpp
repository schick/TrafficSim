//
// Created by maxi on 1/18/19.
//

#include "optimization/RandomOptimizer.h"
#include "AdvanceAlgorithm.h"
#include "optimization/model/OptimizeScenario.h"
#include "optimization/model/SignalLayout.h"

#include <memory>
#include <stdexcept>

nlohmann::json RandomOptimizer::optimize() {

    constexpr size_t iterationsPerRound = 120;

    while (true) {

        iterations += iterationsPerRound;

        nlohmann::json solution;

#pragma omp parallel for
        for (size_t i = 0; i < iterationsPerRound; i++) {

            SignalLayout sigLayout(algorithm, scenarioData);

            if (sigLayout.getTravelledDistance() > minTravelLength) {
                std::lock_guard<std::mutex> lock(solutionLock);
                if (solution.empty())
                    solution = sigLayout.toJson();
            }
        }

        if (!solution.empty())
            return solution;
    }
}
