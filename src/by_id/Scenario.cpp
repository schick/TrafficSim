//
// Created by oke on 16.12.18.
//

//
// Created by oke on 07.12.18.
//

#include <algorithm>

#include "by_id/Scenario.h"

void Scenario::parse(json input) {

    parseJunctions(input);

    parseRoads(input);

    parseCars(input);

    initJunctions();
}

void Scenario::parseJunctions(json &input) {
    junctions.resize(input["junctions"].size());
    int working_id = 0;
    for (const auto& junction : input["junctions"]) {
        double x = junction["x"];
        double y = junction["y"];
        junction_original_to_working_ids[junction["id"]] = working_id;
        junction_working_to_original_ids[working_id] = junction["id"];
        junctions.at(working_id) = Junction(working_id, x * 100.0, y * 100.0,
                                         (int) signals.size(), junction["signals"].size());

        // store signals in this->signal
        for (const auto& signal : junction["signals"]) {
            signals.emplace_back(Junction::Signal({ signal["time"], signal["dir"] }));
        }
        working_id++;
    }
}

// helper to calculate direction of a road
Junction::Direction calcDirectionOfRoad(Junction &from, Junction &to) {
    // linkshändisches koordinatensystemv
    if (from.y < to.y) {
        return Junction::Direction::SOUTH;
    } else if (from.y > to.y) {
        return Junction::Direction::NORTH;
    } else if (from.x < to.x) {
        return Junction::Direction::EAST;
    } else if (from.x > to.x) {
        return Junction::Direction::WEST;
    }
    printf("ERROR: not a valid road...");
    exit(-1);
}

void Scenario::parseRoads(json &input) {
    roads.resize(2 * input["roads"].size());
    int road_id = 0;
    int lane_id = 0;

    for (const auto& road : input["roads"]) {
        /* one for each direction */
        for (int j = 0; j < 2; j++) {
            int from, to;
            if (j == 0) {
                from = road["junction1"];
                to = road["junction2"];
            }
            else {
                from = road["junction2"];
                to = road["junction1"];
            }

            double roadLimit = static_cast<double>(road["limit"]) / 3.6;
            double length = fabs(junctions.at(from).x - junctions.at(to).x) + fabs(junctions.at(from).y - junctions.at(to).y);
            Junction::Direction roadDir = calcDirectionOfRoad(junctions.at(from), junctions.at(to));
            roads.at(road_id) = Road(road_id, from, to, roadLimit, length, roadDir);

            // create lanes
            for (uint8_t lane_num = 0; lane_num < road["lanes"]; lane_num++) {
                lanes.emplace_back(Lane(lane_num, road_id, lane_id, length));
                roads.at(road_id).lanes.at(lane_num) = lane_id;
                lane_id++;
            }

            junctions.at(from).outgoing.at(roadDir) = road_id;
            junctions.at(to).incoming.at((roadDir + 2) % 4) = road_id;
            road_id++;
        }
    }
}

void Scenario::parseCars(json &input) {
    cars.resize(input["cars"].size());
    int new_car_id = 0;
    for (const auto& car : input["cars"]) {
        double target_velocity = static_cast<double>(car["target_velocity"]) / 3.6;
        car_original_to_working_ids[car["id"]] = new_car_id;

        int from_id = junction_original_to_working_ids[car["start"]["from"]];
        int to_id = junction_original_to_working_ids[car["start"]["to"]];
        auto it = std::find_if(std::begin(roads), std::end(roads), [&](const Road &road) {
            return (road.from == from_id && road.to == to_id); });
        assert(it != roads.end());
        int lane_id = (*it).lanes.at(car["start"]["lane"]);
        assert(lane_id != -1);
        Car &car_obj = cars[new_car_id] = Car(
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

void Scenario::initJunctions() {
    int red_traffic_id = (int) cars.size();
    for (auto &junction : junctions) {
        for(int i = 0; i < 4; i++) {
            if (junction.incoming.at(i) != -1) {
                Road &road = roads.at(junction.incoming.at(i));
                int j = 0;
                for(; j < road.lanes.size(); j++) {
                    traffic_lights.emplace_back(RedTrafficLight(red_traffic_id, road.lanes.at(j), road.length - 35. / 2.));
                    junction.red_traffic_lights_ids.at(i).at(j) = red_traffic_id;
                    red_traffic_id++;
                }
                for(;j < 3; j++) {
                    junction.red_traffic_lights_ids.at(i).at(j) = -1;
                }
            } else {
                junction.red_traffic_lights_ids.at(i).fill(-1);
            }
        }
        junction.initializeSignals(*this);
    }
}

json Scenario::toJson() {
    json output;
    std::vector<int> car_ids;
    for(auto &a : car_original_to_working_ids)
        car_ids.emplace_back(a.first);

    std::sort(car_ids.begin(), car_ids.end());

    for (int id : car_ids) {
        Car &car = cars.at(car_original_to_working_ids[id]);
        json out_car;
        Lane &l = lanes.at(car.getLane());
        Road &r = roads.at(l.road);
        out_car["id"] = id;
        out_car["from"] = junction_working_to_original_ids[r.from];
        out_car["to"] = r.to;
        out_car["lane"] = l.lane_num;
        out_car["position"] = car.getPosition();

        output["cars"].push_back(out_car);
    }
    return output;
}
