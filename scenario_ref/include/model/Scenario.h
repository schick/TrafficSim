//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H

#include <memory>
#include <BaseScenario.h>

#include "util/json_fwd.hpp"

#include "BaseScenario.h"
#include "Car.h"
#include "Junction.h"
#include "Lane.h"
#include "RedTrafficLight.h"
#include "Road.h"

using json = nlohmann::json;

class Scenario : public BaseScenario {
public:

    std::vector<std::shared_ptr<Junction>> junctions;
    std::vector<std::shared_ptr<Road>> roads;
    std::vector<std::shared_ptr<Lane>> lanes;
    std::vector<std::shared_ptr<Car>> cars;

    void parse(json input);
    json toJson();

    void parseJunctions(json &input);
    void parseRoads(json & input);
    void createRoads(const nlohmann::json & road);
    Junction::Direction calcDirectionOfRoad(Junction *from, Junction *to);
    void createLanesForRoad(const nlohmann::json & road, std::shared_ptr<Road> &road_obj);
    void parseCars(json & input);
    void initJunctions();

};


#endif //PROJECT_SCENARIO_H