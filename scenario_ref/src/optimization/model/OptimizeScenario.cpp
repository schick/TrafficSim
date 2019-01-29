//
// Created by maxi on 1/18/19.
//

#include "optimization/model/OptimizeScenario.h"

void OptimizeScenario::parse(nlohmann::json &input) {
    parsedJson = input;
    parseJunctions(input);
    parseRoads(input);
    parseCars(input);
    total_steps = input["time_steps"];
    current_step = 0;
}

json OptimizeScenario::toJson() {
    json output;
    for (Junction &junction : junctions) {
        json out_junction;

        out_junction["id"] = junction.id;

        for (Junction::Signal &signal : junction.signals) {
            json out_signal;

            out_signal["dir"] = static_cast<int>(signal.direction);
            out_signal["time"] = signal.duration;

            out_junction["signals"].push_back(out_signal);
        }

        output["junctions"].push_back(out_junction);
    }
    return output;
}

void OptimizeScenario::parseSignals(const json &input) {
    //Do nothing
}

void OptimizeScenario::reset() {
    size_t idx = 0;
    for (const auto &car : parsedJson["cars"]) {

        Car &car_instance = this->cars[idx];

        car_instance.v = 0;
        car_instance.x = car["start"]["distance"];

        uint64_t fromID = car["start"]["from"];
        uint64_t toID = car["start"]["to"];
        Junction *from = junctionsMap.at(fromID);
        Junction *to = junctionsMap.at(toID);

        auto roadDir = calcDirectionOfRoad(from, to);
        uint8_t startLaneIndex = car["start"]["lane"];

        auto lane = from->outgoing[roadDir]->lanes[startLaneIndex];
        car_instance.moveToLane(*lane);

        car_instance.turns.clear();
        for (const auto &route : car["route"])
            car_instance.turns.emplace_back(route);

        car_instance.travelledDistance = 0;
        car_instance.a = 0;
        car_instance.advance_data.new_acceleration = 0;
        car_instance.advance_data.new_lane_offset = 0;
        car_instance.advance_data.sameLaneAcceleration = 0;
        car_instance.advance_data.rightLaneAcceleration = 0;
        car_instance.advance_data.leftLaneAcceleration = 0;
        idx++;
    }
    current_step = 0;
}