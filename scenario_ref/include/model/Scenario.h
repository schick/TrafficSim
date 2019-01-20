//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H


#include <memory>
#include <unordered_map>
#include <cmath>
#include <BaseScenario.h>

#include "util/json.hpp"

#include "BaseScenario.h"
#include "Car.h"
#include "Junction.h"
#include "Lane.h"
#include "TrafficLight.h"
#include "Road.h"

using json = nlohmann::json;

class Scenario : public BaseScenario {
public:

    //TODO: Set hash algorithm to just take the id

    // Needed for faster parsing
    std::unordered_map<uint64_t, Junction*> junctionsMap;

    std::vector<Junction> junctions;
    std::vector<Road> roads;
    std::vector<Lane> lanes;
    std::vector<Car> cars;
    std::vector<TrafficLight> trafficLights;

    void parse(json input);
    json toJson();

    void parseJunctions(json &input);
    void parseRoads(json &input);
    double calcRoadLength(Junction* from, Junction *to);
    void createRoad(Junction* from, Junction *to, double roadLength, double speedLimit, uint8_t laneCount);
    Junction::Direction calcDirectionOfRoad(Junction *from, Junction *to);
    void createLanesForRoad(uint8_t  laneCount, double roadLength, Road &road_obj);
    void parseCars(json &input);
    void initJunctions();

};


#endif //PROJECT_SCENARIO_H