//
// Created by maxi on 1/19/19.
//

#ifndef TRAFFIC_SIM_BASEOPTIMIZER_H
#define TRAFFIC_SIM_BASEOPTIMIZER_H


#include "util/json.hpp"

class BaseOptimizer {

public:

    BaseOptimizer(nlohmann::json &scenarioData, std::string &algorithm): scenarioData(scenarioData), algorithm(algorithm) {
        minTravelLength = scenarioData["min_travel_distance"];
    }

    virtual nlohmann::json optimize() = 0;

protected:

    nlohmann::json &scenarioData;
    std::string &algorithm;

    double minTravelLength;

};


#endif //TRAFFIC_SIM_BASEOPTIMIZER_H
