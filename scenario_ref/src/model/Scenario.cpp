//
// Created by oke on 07.12.18.
//

#include "model/Scenario.h"

//TODO: Rework so that no temporary instances are needed

void Scenario::parse(json input) {

    parseJunctions(input);

    parseRoads(input);

    parseCars(input);

    initJunctions();
}

//JUNCTIONS
void Scenario::parseJunctions(json &input) {
    //Set correct size of hash map
    junctions.reserve(input["junctions"].size());

    for (const auto &junction : input["junctions"]) {
        uint64_t id = junction["id"];
        double x = static_cast<double>(junction["x"]) * 100.0;
        double y = static_cast<double>(junction["y"]) * 100.0;
        std::shared_ptr<Junction> junction_obj = std::make_shared<Junction>(id, x, y);

        for (const auto &signal : junction["signals"]) {
            junction_obj->signals.emplace_back(signal["time"], signal["dir"]);
        }

        junctions.insert({id, std::move(junction_obj)});
    }
}

//ROADS
void Scenario::parseRoads(json &input) {
    //Set correct size of vector
    roads.reserve(input["roads"].size() * 2);

    for (const auto &road : input["roads"]) {
        uint64_t j1_id = road["junction1"];
        uint64_t j2_id = road["junction2"];

        Junction *junction1 = junctions.at(j1_id).get();
        Junction *junction2 = junctions.at(j2_id).get();

        double roadSpeedLimit = static_cast<double>(road["limit"]) / 3.6;
        uint8_t laneCount = road["lanes"];
        double roadLength = calcRoadLength(junction1, junction2);

        /* one for each direction */
        createRoad(junction1, junction2, roadLength, roadSpeedLimit, laneCount);
        createRoad(junction2, junction1, roadLength, roadSpeedLimit, laneCount);
    }
}

double Scenario::calcRoadLength(Junction *junction1, Junction *junction2) {
    return (std::abs(junction1->x - junction2->x) + std::abs(junction1->y - junction2->y));
}


void Scenario::createRoad(Junction *from, Junction *to, double roadLength, double speedLimit, uint8_t laneCount) {
    Junction::Direction roadDir = calcDirectionOfRoad(from, to);

    roads.emplace_back(from, to, speedLimit, roadDir);

    createLanesForRoad(laneCount, roadLength, roads.back());

    from->outgoing[roadDir] = &(roads.back());
    to->incoming[(roadDir + 2) % 4] = &(roads.back());

}

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
    } else {
        printf("ERROR: not a valid road...");
        exit(-1);
    }
}

void Scenario::createLanesForRoad(uint8_t laneCount, double roadLength, Road &road_obj) {
    for (uint8_t lane_id = 0; lane_id < laneCount; lane_id++) {
        std::shared_ptr<Lane> lane = std::make_shared<Lane>(lane_id, &road_obj, roadLength);
        road_obj.lanes.emplace_back(lane.get());
        lanes.emplace_back(std::move(lane));
    }
}

//CARS
void Scenario::parseCars(json &input) {
    cars.reserve(input["cars"].size());

    for (const auto &car : input["cars"]) {
        double target_velocity = static_cast<double>(car["target_velocity"]) / 3.6;
        cars.emplace_back(car["id"],
                    5.,
                    target_velocity,
                    car["max_acceleration"],
                    car["target_deceleration"],
                    car["min_distance"],
                    car["target_headway"],
                    car["politeness"],
                    car["start"]["distance"]);

        uint64_t fromID = car["start"]["from"];
        uint64_t toID = car["start"]["to"];
        auto from = junctions.at(fromID);
        auto to = junctions.at(toID);

        auto roadDir = calcDirectionOfRoad(from.get(), to.get());
        uint8_t startLaneIndex = car["start"]["lane"];

        auto lane = from->outgoing[roadDir]->lanes[startLaneIndex];

        cars.back().moveToLane(lane);

        for (auto &route : car["route"])
            cars.back().turns.emplace_back(route);
    }
}

//INITIALIZE JUNCTIONS
void Scenario::initJunctions() {
    for (auto pair : junctions) {
        pair.second->initializeSignals();
    }
}

//TO JSON
json Scenario::toJson() {
    json output;
    for (Car &car : cars) {
        json out_car;

        out_car["id"] = car.id;
        out_car["from"] = car.getLane()->road->from->id;
        out_car["to"] = car.getLane()->road->to->id;
        out_car["lane"] = car.getLane()->lane;
        out_car["position"] = car.getPosition();

        output["cars"].push_back(out_car);
    }
    return output;
}
