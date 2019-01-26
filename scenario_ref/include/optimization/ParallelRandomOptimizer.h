//
// Created by maxi on 1/18/19.
//

#ifndef TRAFFIC_SIM_RANDOMOPTIMIZER_H
#define TRAFFIC_SIM_RANDOMOPTIMIZER_H


#include "optimization/BaseOptimizer.h"
#include <mutex>

class ParallelRandomOptimizer : public BaseOptimizer {

private:

    size_t iterations = 0;

    std::mutex solutionLock;

public:

    ParallelRandomOptimizer(nlohmann::json &scenarioData, const std::string &algorithm) : BaseOptimizer(scenarioData, algorithm) {}

    nlohmann::json optimize() override;

    size_t getIterations() {
        return iterations;
    }

};


#endif //TRAFFIC_SIM_RANDOMOPTIMIZER_H
