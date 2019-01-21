//
// Created by maxi on 1/18/19.
//

#ifndef TRAFFIC_SIM_RANDOMOPTIMIZER_H
#define TRAFFIC_SIM_RANDOMOPTIMIZER_H


#include "optimization/BaseOptimizer.h"

class RandomOptimizer : public BaseOptimizer {

public:

    RandomOptimizer(nlohmann::json &scenarioData, const std::string &algorithm) : BaseOptimizer(scenarioData, algorithm) {}

    nlohmann::json optimize() override;

};


#endif //TRAFFIC_SIM_RANDOMOPTIMIZER_H
