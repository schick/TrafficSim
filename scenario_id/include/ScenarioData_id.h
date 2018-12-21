//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_DATA_ID_H
#define PROJECT_SCENARIO_DATA_ID_H

#include <memory>
#include <unordered_map>

#include "Car_id.h"
#include "Junction_id.h"
#include "Lane_id.h"
#include "RedTrafficLight_id.h"
#include "Road_id.h"

class ScenarioData_id {

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
    std::unordered_map<int, int> car_original_to_working_ids;
    std::vector<Car_id> cars;
    std::vector<Car_id::TurnDirection> turns;

    inline TrafficObject_id &getTrafficObject(int id) {
        if (id < cars.size())
            return cars[id];
        else
            return traffic_lights[id - cars.size()];
    }
};





#endif //PROJECT_SCENARIO_H
