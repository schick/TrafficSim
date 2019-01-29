//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H

#include <unordered_map>
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

    // Main parse function
    void parse(json &input) override;

    // Return cars as json
    json toJson() override;

    // Set RedTrafficLights depending on set Signals
    void initJunctions();

    // Get sum of travelledDistance of all cars
    double getTravelledDistance() override;

    std::vector<Junction> junctions;
    std::vector<Road> roads;
    std::vector<Lane> lanes;
    std::vector<Car> cars;
    size_t current_step;
    size_t total_steps;

protected:

    // Overridable method that handles parsing of Signals
    virtual void parseSignals(const json &input);

    // Needed for faster parsing
    std::unordered_map<uint64_t, Junction*> junctionsMap;

    // Parse functions
    void parseJunctions(json &input);
    void parseRoads(json &input);
    void createRoad(Junction* from, Junction *to, double roadLength, double speedLimit, uint8_t laneCount);
    void createLanesForRoad(uint8_t  laneCount, double roadLength, Road &road_obj);
    void parseCars(json &input);

    // Helper methods
    Junction::Direction calcDirectionOfRoad(Junction *from, Junction *to);
    double calcRoadLength(Junction* from, Junction *to);

};

#endif //PROJECT_SCENARIO_H