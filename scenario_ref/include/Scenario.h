//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H

#include <memory>
#include <BaseScenario.h>

#include "util/json.hpp"

#include "BaseScenario.h"
#include "Car.h"
#include "Junction.h"
#include "Lane.h"
#include "RedTrafficLight.h"
#include "Road.h"

using json = nlohmann::json;

class Scenario : public BaseScenario {
public:

    std::vector<std::unique_ptr<Junction>> junctions;
    std::vector<std::unique_ptr<Road>> roads;
    std::vector<std::unique_ptr<Lane>> lanes;
    std::vector<std::unique_ptr<Car>> cars;

    void parse(json input);
    json toJson();

    void initJunctions();
    void parseCars(json & input);
    void parseRoads(json & input);
    void createRoads(const nlohmann::json & road);
    void createLanesForRoad(const nlohmann::json & road, std::unique_ptr<Road> &road_obj);
    void parseJunctions(json &input);

};





#endif //PROJECT_SCENARIO_H
