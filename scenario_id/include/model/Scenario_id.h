
#ifndef PROJECT_SCENARIO_ID_H
#define PROJECT_SCENARIO_ID_H

#include <unordered_map>

#include "model/Car_id.h"
#include "model/Junction_id.h"
#include "model/Lane_id.h"
#include "model/RedTrafficLight_id.h"
#include "model/Road_id.h"

#include "BaseScenario.h"

using json = nlohmann::json;

class Scenario_id : public BaseScenario {

public:
    // junctions.
    std::unordered_map<size_t , size_t> junction_working_to_original_ids;
    std::unordered_map<size_t, size_t> junction_original_to_working_ids;
    std::vector<Junction_id> junctions;
    std::vector<Junction_id::Signal> signals;
    std::vector<RedTrafficLight_id> traffic_lights;

    // roads
    std::vector<Road_id> roads;
    std::vector<Lane_id> lanes;

    // cars
    std::unordered_map<size_t, size_t> car_original_to_working_ids;
    std::unordered_map<size_t, size_t> car_working_to_original_ids;
    std::vector<Car_id> cars;
    std::vector<Car_id::TurnDirection> turns;

    void parse(json &input);
    void initJunctions();
    void parseCars(json &input);
    void parseRoads(json &input);
    void parseJunctions(json &input);
    json toJson();

    double getTravelledDistance() override {
        double distance = 0;
        for(auto &car : cars)
            distance += car.travelled_distance;
        return distance;
    }
private:
    void initializeSignals(Junction_id &junction);
    void setSignals(Junction_id &junction);

};

#endif