//
// Created by maxi on 1/17/19.
//

#ifndef TRAFFIC_SIM_CALCULATION_H
#define TRAFFIC_SIM_CALCULATION_H

#include "util/json.hpp"
#include "util/SimpleArgumentParser.h"

namespace trafficSim {

    void calculate(nlohmann::json &input, SimpleArgumentParser &p);

    void optimize(nlohmann::json &input, SimpleArgumentParser &p);

};


#endif //TRAFFIC_SIM_CALCULATION_H
