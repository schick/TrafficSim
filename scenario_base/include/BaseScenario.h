//
// Created by oke on 17.12.18.
//

#ifndef TRAFFIC_SIM_BASESCENARIO_H
#define TRAFFIC_SIM_BASESCENARIO_H

#include "util/json_fwd.hpp"

using json = nlohmann::json;

class BaseScenario {

public:
    virtual void parse(json &input) = 0;
    virtual json toJson() = 0;

};


#endif //TRAFFIC_SIM_BASESCENARIO_H
