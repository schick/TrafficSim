//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H


#include <memory>
#include <unordered_map>
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

    //TODO: Set hash algorithm to just take the id
    std::unordered_map<uint64_t , std::shared_ptr<Junction>> junctions;
    std::vector<std::shared_ptr<Road>> roads;
    std::vector<std::shared_ptr<Lane>> lanes;
    std::vector<std::shared_ptr<Car>> cars;

    void parse(json input);
    json toJson();

    void parseJunctions(json &input);
    void parseRoads(json &input);
    void createRoad(Junction* from, Junction *to, double speedLimit, uint8_t laneCount);
    Junction::Direction calcDirectionOfRoad(Junction *from, Junction *to);
    void createLanesForRoad(uint8_t  laneCount, std::shared_ptr<Road> &road_obj);
    void parseCars(json &input);
    void initJunctions();

};


#endif //PROJECT_SCENARIO_H