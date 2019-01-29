//
// Created by maxi on 1/27/19.
//

#ifndef TRAFFIC_SIM_SEQUENTIALRANDOMOPTIMIZER_H
#define TRAFFIC_SIM_SEQUENTIALRANDOMOPTIMIZER_H


#include "optimization/BaseOptimizer.h"
#include <mutex>

class SequentialRandomOptimizer : public BaseOptimizer {

private:

    size_t iterations = 0;

public:

    SequentialRandomOptimizer(nlohmann::json &scenarioData, const std::string &algorithm) : BaseOptimizer(scenarioData, algorithm) {}

    nlohmann::json optimize() override;

    size_t getIterations() {
        return iterations;
    }

};


#endif //TRAFFIC_SIM_SEQUENTIALRANDOMOPTIMIZER_H
