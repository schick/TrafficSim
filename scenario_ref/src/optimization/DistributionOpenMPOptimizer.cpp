//
// Created by maxi on 1/18/19.
//

#include "optimization/DistributionOpenMPOptimizer.h"

nlohmann::json DistributionOpenMPOptimizer::optimize() {

    std::vector<std::array<double, 4>> incoming_counts = initialSimulation();

    #pragma omp parallel
    {
        randomTestsUntilDone(incoming_counts);
    };

    return validResults.front();
}
