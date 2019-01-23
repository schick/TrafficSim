//
// Created by maxi on 1/20/19.
//

#ifndef TRAFFIC_SIM_SIGNALSLAYOUT_H
#define TRAFFIC_SIM_SIGNALSLAYOUT_H


#include <unordered_map>
#include <model/Junction.h>
#include "OptimizeScenario.h"
#include "util/json.hpp"

class SignalLayout {

public:

    SignalLayout(std::string algorithm, nlohmann::json scenarioData);

    double getTravelledDistance() { return travelledDistance; }

    nlohmann::json toJson();

private:

    std::unordered_map<uint64_t, std::vector<Junction::Signal>> signalsMap;

    double travelledDistance = 0.0;

    void createRandomSignal(Junction &junction);

    void populate(OptimizeScenario &scenario);

};


#endif //TRAFFIC_SIM_SIGNALSLAYOUT_H
