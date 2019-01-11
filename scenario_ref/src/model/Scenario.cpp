//
// Created by oke on 07.12.18.
//

#include "model/Scenario.h"
#include "util/json.hpp"

void Scenario::parse(json input) {

    parseJunctions(input);

    parseRoads(input);

    parseCars(input);

    initJunctions();
}

//JUNCTIONS
void Scenario::parseJunctions(json &input) {
    for (const auto& junction : input["junctions"]) {

        double x = (int16_t) junction["x"] * 100;
        double y = (int16_t) junction["y"] * 100;
        std::shared_ptr<Junction> junction_obj = std::make_shared<Junction>(junction["id"], x * 100.0, y * 100.0);
        for (const auto& signal : junction["signals"]) {
            junction_obj->signals.emplace_back(Junction::Signal({ signal["time"], signal["dir"] }));
        }
        junctions.emplace_back(std::move(junction_obj));
    }
}

void Scenario::parseRoads(json &input) {
    for (const auto& road : input["roads"]) {
        createRoads(road);
    }
}

//ROADS
void Scenario::createRoads(const nlohmann::json & road) {
    uint64_t j1 = road["junction1"];
    uint64_t j2 = road["junction2"];
    auto junction1 = std::find_if(std::begin(junctions), std::end(junctions),
                                  [&](const std::shared_ptr<Junction> &v) { return v->id == j1; });
    auto junction2 = std::find_if(std::begin(junctions), std::end(junctions),
                                  [&](const std::shared_ptr<Junction> &v) { return v->id == j2; });

    /* one for each direction */
    for (int j = 0; j < 2; j++) {
        Junction *from, *to;
        // get first junction
        assert(junction1 != std::end(junctions));
        assert(junction2 != std::end(junctions));
        if (j == 0) {
            from = (*junction1).get();
            to = (*junction2).get();
        }
        else {
            from = (*junction2).get();
            to = (*junction1).get();
        }

        double roadLimit = static_cast<double>(road["limit"]) / 3.6;
        Junction::Direction roadDir = calcDirectionOfRoad(from, to);

        std::shared_ptr<Road> road_obj = std::make_shared<Road>(from, to, roadLimit, roadDir);

        createLanesForRoad(road, road_obj);

        road_obj->from->outgoing[roadDir] = road_obj.get();
        road_obj->to->incoming[(roadDir + 2) % 4] = road_obj.get();

        roads.emplace_back(std::move(road_obj));
    }
}

// helper to calculate direction of a road
Junction::Direction Scenario::calcDirectionOfRoad(Junction *from, Junction *to) {
    // linkshändisches koordinatensystem
    if (from->y < to->y) {
        return Junction::Direction::SOUTH;
    } else if (from->y > to->y) {
        return Junction::Direction::NORTH;
    } else if (from->x < to->x) {
        return Junction::Direction::EAST;
    } else if (from->x > to->x) {
        return Junction::Direction::WEST;
    }
    printf("ERROR: not a valid road...");
    exit(-1);
}

void Scenario::createLanesForRoad(const nlohmann::json & road, std::shared_ptr<Road> &road_obj)
{
    for (uint8_t lane_id = 0; lane_id < road["lanes"]; lane_id++) {
        std::shared_ptr<Lane> lane = std::make_shared<Lane>(lane_id, road_obj.get());
        road_obj->lanes.emplace_back(lane.get());
        lanes.emplace_back(std::move(lane));
    }
}

//CARS
void Scenario::parseCars(json &input) {
    cars.resize(input["cars"].size());
    int car_idx = 0;
    for (const auto& car : input["cars"]) {
        double target_velocity = static_cast<double>(car["target_velocity"]) / 3.6;
        cars[car_idx] = std::make_shared<Car>(
                car["id"],
                5.,
                target_velocity,
                car["max_acceleration"],
                car["target_deceleration"],
                car["min_distance"],
                car["target_headway"],
                car["politeness"],
                car["start"]["distance"]);

        uint64_t from = car["start"]["from"];
        uint64_t to = car["start"]["to"];
        auto it = std::find_if(std::begin(roads), std::end(roads), [&](const std::shared_ptr<Road> &road) {
            return ((road->from->id == from && road->to->id == to)); });
        assert(it != roads.end());

        cars[car_idx]->moveToLane((*it)->lanes[car["start"]["lane"]]);

        for (const auto& route : car["route"]) cars[car_idx]->turns.push_back(route);

        car_idx++;
    }
}

//INITIALIZE JUNCTIONS
void Scenario::initJunctions() {
    for (std::shared_ptr<Junction> &j : junctions) {
        j->initializeSignals();
    }
}

//TO JSON
json Scenario::toJson() {
    json output;
    for (const auto& car : cars) {
        json out_car;

        out_car["id"] = car->id;
        out_car["from"] = car->getLane()->road->from->id;
        out_car["to"] = car->getLane()->road->to->id;
        out_car["lane"] = car->getLane()->lane;
        out_car["position"] = car->getPosition();

        output["cars"].push_back(out_car);
    }
    return output;
}
