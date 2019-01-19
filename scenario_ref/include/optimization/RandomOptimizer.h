//
// Created by maxi on 1/18/19.
//

#ifndef TRAFFIC_SIM_RANDOMOPTIMIZER_H
#define TRAFFIC_SIM_RANDOMOPTIMIZER_H

#include "util/json.hpp"


class RandomOptimizer {

public:

    RandomOptimizer(nlohmann::json &scenarioData, std::string &algorithm)
            : scenarioData(scenarioData), algorithm(algorithm) {
        minTravelLength = scenarioData["min_travel_distance"];
    }


    nlohmann::json optimize();


private:

    nlohmann::json &scenarioData;
    std::string &algorithm;

    double minTravelLength = 0.0;

};


#endif //TRAFFIC_SIM_RANDOMOPTIMIZER_H
