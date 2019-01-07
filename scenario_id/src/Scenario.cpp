//
// Created by oke on 16.12.18.
//

//
// Created by oke on 07.12.18.
//

#include <algorithm>
#include "util/json.hpp"
#include "Scenario_id.h"

void Scenario_id::parse(json input) {

    parseJunctions(input);

    parseRoads(input);

    parseCars(input);

    initJunctions();
}

void Scenario_id::parseJunctions(json &input) {
    junctions.resize(input["junctions"].size());
    int working_id = 0;
    for (const auto& junction : input["junctions"]) {
        double x = junction["x"];
        double y = junction["y"];
        junction_original_to_working_ids[junction["id"]] = working_id;
        junction_working_to_original_ids[working_id] = junction["id"];
        junctions.at(working_id) = Junction_id(working_id, x * 100.0, y * 100.0,
                                         (int) signals.size(), junction["signals"].size());

        // store signals in this->signal
        for (const auto& signal : junction["signals"]) {
            signals.emplace_back(Junction_id::Signal({ signal["time"], signal["dir"] }));
        }
        working_id++;
    }
}

// helper to calculate direction of a road
Junction_id::Direction calcDirectionOfRoad(Junction_id &from, Junction_id &to) {
    // linkshändisches koordinatensystemv
    if (from.y < to.y) {
        return Junction_id::Direction::SOUTH;
    } else if (from.y > to.y) {
        return Junction_id::Direction::NORTH;
    } else if (from.x < to.x) {
        return Junction_id::Direction::EAST;
    } else if (from.x > to.x) {
        return Junction_id::Direction::WEST;
    }
    printf("ERROR: not a valid road...");
    exit(-1);
}

void Scenario_id::parseRoads(json &input) {
    roads.resize(2 * input["roads"].size());
    size_t road_id = 0;
    size_t lane_id = 0;

    for (const auto& road : input["roads"]) {
        /* one for each direction */
        for (int j = 0; j < 2; j++) {
            size_t from, to;
            if (j == 0) {
                from = junction_original_to_working_ids[road["junction1"]];
                to = junction_original_to_working_ids[road["junction2"]];
            }
            else {
                from = junction_original_to_working_ids[road["junction2"]];
                to = junction_original_to_working_ids[road["junction1"]];
            }
            double roadLimit = static_cast<double>(road["limit"]) / 3.6;
            double length = fabs(junctions.at(from).x - junctions.at(to).x) + fabs(junctions.at(from).y - junctions.at(to).y);
            Junction_id::Direction roadDir = calcDirectionOfRoad(junctions.at(from), junctions.at(to));
            roads.at(road_id) = Road_id(road_id, from, to, roadLimit, length, roadDir);

            // create lanes
            for (uint8_t lane_num = 0; lane_num < road["lanes"]; lane_num++) {
                lanes.emplace_back(Lane_id(lane_num, road_id, lane_id, length));
                roads.at(road_id).lanes[lane_num] = lane_id;
                lane_id++;
            }

            junctions.at(from).outgoing[roadDir] = road_id;
            junctions.at(to).incoming[(roadDir + 2) % 4] = road_id;
            road_id++;
        }
    }
}

void Scenario_id::parseCars(json &input) {
    cars.resize(input["cars"].size());
    int new_car_id = 0;
    for (const auto& car : input["cars"]) {
        double target_velocity = static_cast<double>(car["target_velocity"]) / 3.6;
        car_original_to_working_ids[car["id"]] = new_car_id;
        car_working_to_original_ids[new_car_id] = car["id"];

        int from_id = junction_original_to_working_ids[car["start"]["from"]];
        int to_id = junction_original_to_working_ids[car["start"]["to"]];
        auto it = std::find_if(std::begin(roads), std::end(roads), [&](const Road_id &road) {
            return (road.from == from_id && road.to == to_id); });
        assert(it != roads.end());
        int lane_id = (*it).lanes[(int) car["start"]["lane"]];
        assert(lane_id != -1);
        Car_id &car_obj = cars[new_car_id] = Car_id(
                new_car_id,
                5.,
                target_velocity,
                car["max_acceleration"],
                car["target_deceleration"],
                car["min_distance"],
                car["target_headway"],
                car["politeness"],
                (int) turns.size(), car["route"].size(),
                lane_id, car["start"]["distance"]);

        for (const auto& route : car["route"]) turns.push_back(route);
        new_car_id++;
    }
}

void Scenario_id::initJunctions() {
    size_t red_traffic_id = (int) cars.size();
    for (auto &junction : junctions) {
        for(int i = 0; i < 4; i++) {
            if (junction.incoming[i] != -1) {
                Road_id &road = roads.at(junction.incoming[i]);
                int j = 0;
                for(; j < 3; j++) {
                    if (road.lanes[j] != -1) {
                        traffic_lights.emplace_back(RedTrafficLight_id(red_traffic_id, road.lanes[j], road.length - 35. / 2.));
                        junction.red_traffic_lights_ids[i][j] = red_traffic_id;
                        red_traffic_id++;
                    } else {
                        junction.red_traffic_lights_ids[i][j] = -1;
                    }
                }
                for(;j < 3; j++) {
                    junction.red_traffic_lights_ids[i][j] = -1;
                }
            } else {
                for(auto &i : junction.red_traffic_lights_ids[i]) i = (size_t ) -1;
            }
        }
        junction.initializeSignals(*this);
    }
}

json Scenario_id::toJson() {
    json output;
    std::vector<int> car_ids;
    for(auto &a : car_original_to_working_ids)
        car_ids.emplace_back(a.first);

    std::sort(car_ids.begin(), car_ids.end());

    for (int id : car_ids) {
        Car_id &car = cars.at(car_original_to_working_ids[id]);
        json out_car;
        Lane_id &l = lanes.at(car.getLane());
        Road_id &r = roads.at(l.road);
        out_car["id"] = id;
        out_car["from"] = junction_working_to_original_ids[r.from];
        out_car["to"] = r.to;
        out_car["lane"] = l.lane_num;
        out_car["position"] = car.getPosition();

        output["cars"].push_back(out_car);
    }
    return output;
}
