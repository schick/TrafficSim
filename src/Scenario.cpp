//
// Created by oke on 07.12.18.
//

#include "Scenario.h"
void Scenario::parse(json input) {

    for(const auto& junction : input["junctions"])
    {
        double x = junction["x"];
        double y = junction["y"];
        std::unique_ptr<Junction> junction_obj = std::make_unique<Junction>(junction["id"], x * 100, y * 100);
        for(const auto& signal : junction["signals"])
            junction_obj->signals.emplace_back(Junction::Signal({signal["dir"], signal["time"]}));
        junctions.emplace_back(std::move(junction_obj));
    }

    for(const auto& road : input["roads"])
    {
        /* one for each direction */
        for(int j = 0; j < 2; j++) {
            Junction *from, *to;
            // get first junction
            auto it = std::find_if(std::begin(junctions), std::end(junctions),
                                   [&](const std::unique_ptr<Junction> &v) { return v->id == road["junction1"]; });
            assert(it != std::end(junctions));
            if (j == 0)
                from = (*it).get();
            else
                to = (*it).get();

            // get second junction
            it = std::find_if(std::begin(junctions), std::end(junctions),
                              [&](const std::unique_ptr<Junction> &v) { return v->id == road["junction2"]; });
            assert(it != std::end(junctions));
            if (j == 1)
                from = (*it).get();
            else
                to = (*it).get();

            std::unique_ptr<Road> road_obj = std::make_unique<Road>(from, to, road["limit"]);

            for (uint8_t lane_id = 0; lane_id < road["lanes"]; lane_id++) {
                std::unique_ptr<Lane> lane = std::make_unique<Lane>(lane_id, road_obj.get());
                road_obj->lanes.emplace_back(lane.get());
                lanes.emplace_back(std::move(lane));
            }

            Road *ptr = road_obj.get();
            if (road_obj->from->x < road_obj->to->x) {
                road_obj->from->outgoing[Junction::Direction::EAST] = ptr;
                road_obj->to->incoming[Junction::Direction::WEST] = ptr;
            } else if (road_obj->from->x > road_obj->to->x) {
                road_obj->from->outgoing[Junction::Direction::WEST] = ptr;
                road_obj->to->incoming[Junction::Direction::EAST] = ptr;
            } else if (road_obj->from->y < road_obj->to->y) {
                road_obj->from->outgoing[Junction::Direction::NORTH] = ptr;
                road_obj->to->incoming[Junction::Direction::SOUTH] = ptr;
            } else if (road_obj->from->y > road_obj->to->y) {
                road_obj->from->outgoing[Junction::Direction::SOUTH] = ptr;
                road_obj->to->incoming[Junction::Direction::NORTH] = ptr;
            }
            roads.emplace_back(std::move(road_obj));
        }
    }

    for(const auto& car : input["cars"])
    {
        std::unique_ptr<Car> car_obj = std::make_unique<Car>(
                car["id"], 5., car["target_velocity"], car["max_acceleration"], car["target_deceleration"],
                car["min_distance"], car["target_headway"], car["politeness"],
                car["start"]["distance"]);

        auto it = std::find_if(std::begin(roads), std::end(roads), [&](const std::unique_ptr<Road> &road){
            return ((road->from->id == car["start"]["from"] && road->to->id == car["start"]["to"])); } );
        assert(it != roads.end());

        car_obj->moveToLane((*it)->lanes[car["start"]["lane"]]);

        for(const auto& route : car["route"]) car_obj->turns.push_back(route);

        cars.emplace_back(std::move(car_obj));
    }
}

json Scenario::toJson() {
    json output;
    for(const auto& car : cars)
    {
        json out_car;

        out_car["id"] = car->id;
        out_car["from"] = car->getLane()->road->from->id;
        out_car["to"] = car->getLane()->road->to->id;
        out_car["lane"] = car->getLane()->lane_id;
        out_car["position"] = car->x;

        output["cars"].push_back(out_car);
    }
    return output;
}

void AdvanceAlgorithm::advance(size_t steps) {
    for(int i=0; i < steps; i++) {
        advanceCars();
        advanceTrafficLights();
    }
}