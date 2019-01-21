//
// Created by maxi on 1/18/19.
//

#ifndef TRAFFIC_SIM_DISTOMPOPTIMIZER_H
#define TRAFFIC_SIM_DISTOMPOPTIMIZER_H


#include "optimization/DistributionOptimizer.h"
#include <mutex>

class DistributionOpenMPOptimizer : public DistributionOptimizer {

public:

    DistributionOpenMPOptimizer(nlohmann::json &scenarioData, const std::string &algorithm) : DistributionOptimizer(scenarioData, algorithm) {}

    nlohmann::json optimize() override;

};


#endif //TRAFFIC_SIM_RANDOMOPTIMIZER_H
