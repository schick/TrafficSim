//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H

#include <memory>
#include <unordered_map>

#include "by_id/Car.h"
#include "by_id/Junction.h"
#include "by_id/Lane.h"
#include "by_id/RedTrafficLight.h"
#include "by_id/Road.h"

class ScenarioData {

public:
    // junctions.
    std::unordered_map<int, int> junction_working_to_original_ids;
    std::unordered_map<int, int> junction_original_to_working_ids;
    std::vector<Junction> junctions;
    std::vector<Junction::Signal> signals;
    std::vector<RedTrafficLight> traffic_lights;

    // roads
    std::vector<Road> roads;
    std::vector<Lane> lanes;

    // cars
    std::unordered_map<int, int> car_original_to_working_ids;
    std::vector<Car> cars;
    std::vector<Car::TurnDirection> turns;

    inline TrafficObject &getTrafficObject(int id) {
        if (id < cars.size())
            return cars[id];
        else
            return traffic_lights[id - cars.size()];
    }
};





#endif //PROJECT_SCENARIO_H
