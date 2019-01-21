//
// Created by maxi on 1/18/19.
//

#ifndef TRAFFIC_SIM_RANDOMOPTIMIZER_H
#define TRAFFIC_SIM_RANDOMOPTIMIZER_H


#include "optimization/BaseOptimizer.h"

class RandomOptimizer : public BaseOptimizer {

private:

    int iterations = 0;

public:

    RandomOptimizer(nlohmann::json &scenarioData, const std::string &algorithm) : BaseOptimizer(scenarioData, algorithm) {}

    nlohmann::json optimize() override;

    int getIterations() {
        return iterations;
    }

};


#endif //TRAFFIC_SIM_RANDOMOPTIMIZER_H
