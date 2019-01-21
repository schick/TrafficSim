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

    while (true) {

        iterations += 1200;

        std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
        if (advancer == nullptr) {
            throw std::runtime_error("Algorithm not found: " + algorithm);
        }

        OptimizeScenario *scenario = dynamic_cast<OptimizeScenario *>(advancer->getScenario().get());
        if (scenario == nullptr) {
            throw std::runtime_error("Algorithm '" + algorithm + "' with wrong scenario type for 'RandomOptimizer'");
        }

        std::vector<SignalLayout> signalLayouts;
        signalLayouts.reserve(1200);

        for (size_t i = 0; i < 1200; i++)
            signalLayouts.emplace_back(*scenario);

        nlohmann::json solution;

#pragma omp parallel for
        for (size_t i = 0; i < 1200; i++) {
            std::shared_ptr<AdvanceAlgorithm> itAdvancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
            OptimizeScenario *itScenario = dynamic_cast<OptimizeScenario *>(itAdvancer->getScenario().get());

            signalLayouts[i].populate(*itScenario);
            itAdvancer->advance(scenarioData["time_steps"]);

            if (itScenario->getTravelledDistance() > minTravelLength) {
                std::lock_guard<std::mutex> lock(solutionLock);
                if (solution.empty())
                    solution = itScenario->toJson();
            }
        }

        if (!solution.empty())
            return solution;
    }
}
