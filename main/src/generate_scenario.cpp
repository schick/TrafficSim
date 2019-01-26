#include "memory"
#include <vector>
#include "util/json.hpp"

#include "generate_scenario.h"

int main(int argc, char *argv[]) {


    std::vector<car> cars;
    std::vector<junction> junctions;
    std::vector<road> roads;
    get_random(argc, argv, cars, junctions, roads);
    nlohmann::json output;

    float target_deceleration;
    float max_acceleration;
    float target_headway;
    float politeness;
    int route[4];
    float target_velocity;
    int id;
    float min_distance;
    int from;
    int to;
    int lane;
    int distance;
    auto &json_cars = output["cars"];
    for(auto &c : cars) {
        nlohmann::json json_car;
        json_car["target_deceleration"] = c.target_deceleration;
        json_car["max_acceleration"] = c.max_acceleration;
        json_car["target_headway"] = c.target_headway;
        json_car["politeness"] = c.politeness;
        json_car["target_velocity"] = c.target_velocity;
        json_car["id"] = c.id;
        json_car["min_distance"] = c.min_distance;
        json_car["start"]["from"] = c.from;
        json_car["start"]["to"] = c.to;
        json_car["start"]["lane"] = c.lane;
        json_car["start"]["distance"] = c.distance;

        for(auto &r : c.route) {
            if(r < 4) json_car["route"].push_back(r);
        }
        if(json_car["route"].size() == 0) json_car["route"].push_back(1);
        json_cars.push_back(json_car);
    }
    auto &json_junctions = output["junctions"];
    for(auto &j : junctions) {
        nlohmann::json json_junction;

        json_junction["x"] = j.x;
        json_junction["y"] = j.y;
        json_junction["id"] = j.idx;
        for(auto &s : j.signals) {
            if (s.time > 0) {
                nlohmann::json signal;
                signal["dir"] = s.dir;
                signal["time"] = s.time;
                json_junction["signals"].push_back(signal);
            }
        }
        if (json_junction["signals"].size() == 0) continue;
        json_junctions.push_back(json_junction);
    }


    auto &json_roads = output["roads"];
    for(auto &j : roads) {
        nlohmann::json json_road;

        json_road["lanes"] = j.lanes;
        json_road["junction1"] = j.from_id;
        json_road["junction2"] = j.to_id;
        json_road["limit"] = j.limit;

        json_roads.push_back(json_road);

    }
    output["time_steps"] = 100;
    printf("%s", output.dump().c_str());

}