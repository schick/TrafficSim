//
// Created by maxi on 1/23/19.
//

#ifndef TRAFFIC_SIM_GENETICOPTIMIZER_H
#define TRAFFIC_SIM_GENETICOPTIMIZER_H


#include <optimization/BaseOptimizer.h>
#include <mutex>

class GeneticOptimizer : public BaseOptimizer {

private:

    size_t iterations = 0;

public:

    GeneticOptimizer(nlohmann::json &scenarioData, const std::string &algorithm)
        : BaseOptimizer(scenarioData, algorithm) {}

    nlohmann::json optimize() override;

    size_t getIterations() {
        return iterations;
    }

};


#endif //TRAFFIC_SIM_GENETICOPTIMIZER_H
