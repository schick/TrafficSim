//
// Created by maxi on 1/20/19.
//

#ifndef TRAFFIC_SIM_SIGNALSLAYOUT_H
#define TRAFFIC_SIM_SIGNALSLAYOUT_H


#include <unordered_map>
#include <model/Junction.h>
#include "OptimizeScenario.h"

class SignalLayout {

public:

    explicit SignalLayout(OptimizeScenario &scenario);

    void populate(OptimizeScenario &scenario);

private:

    std::unordered_map<uint64_t, std::vector<Junction::Signal>> signalsMap;

    void createRandomSignal(Junction &junction);

};


#endif //TRAFFIC_SIM_SIGNALSLAYOUT_H
