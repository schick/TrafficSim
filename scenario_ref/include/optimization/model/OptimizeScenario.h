//
// Created by maxi on 1/18/19.
//

#ifndef TRAFFIC_SIM_OPTIMIZESCENARIO_H
#define TRAFFIC_SIM_OPTIMIZESCENARIO_H

#include "model/Scenario.h"

class OptimizeScenario : public Scenario {

public:

    double getTraveledDistance();

    void parse(json &input) override;

private:

    void parseSignals(const json &input) override;

};


#endif //TRAFFIC_SIM_OPTIMIZESCENARIO_H
