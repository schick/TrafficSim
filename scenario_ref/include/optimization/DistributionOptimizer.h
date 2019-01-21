//
// Created by maxi on 1/18/19.
//

#ifndef TRAFFIC_SIM_DISTOPTIMIZER_H
#define TRAFFIC_SIM_DISTOPTIMIZER_H


#include "optimization/BaseOptimizer.h"
#include <mutex>

class DistributionOptimizer : public BaseOptimizer {

public:

    DistributionOptimizer(nlohmann::json &scenarioData, const std::string &algorithm) : BaseOptimizer(scenarioData, algorithm) {}

    nlohmann::json optimize() override;

protected:

    bool IsDone();
    std::vector<std::array<double, 4>> initialSimulation();
    void randomTestsUntilDone(std::vector<std::array<double, 4>> &incoming_counts);

    std::mutex validResultsMutex;
    std::vector<nlohmann::json> validResults;
};


#endif //TRAFFIC_SIM_RANDOMOPTIMIZER_H
