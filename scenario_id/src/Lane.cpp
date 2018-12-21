//
// Created by oke on 07.12.18.
//

#include "Scenario_id.h"

Lane_id::NeighboringObjects Lane_id::getNeighboringObjects(const Scenario_id &s, const TrafficObject_id &object) const {
    // TODO: assert is sorted
    NeighboringObjects result;

    TrafficObject_id::Cmp cmp; //  a < b

    const TrafficObject_id *closest_gt = nullptr;
    const TrafficObject_id *closest_lt = nullptr;

    for(auto &it : s.cars) {
        if(it.getLane() == id && &it != &object) {
            if(cmp(&it, &object)) {
                // objects behind
                if(closest_lt == nullptr || !cmp(&it, closest_lt)) {
                    closest_lt = &it;
                }
            } else {
                // objects in front of
                if(closest_gt == nullptr || cmp(&it, closest_gt)) {
                    closest_gt = &it;
                }
            }
        }
    }

    for(auto &it : s.traffic_lights) {
        if(it.getLane() == id && &it != &object) {
            if(cmp(&it, &object)) {
                // objects behind
                if(closest_lt == nullptr || !cmp(&it, closest_lt)) {
                    closest_lt = &it;
                }
            } else {
                // objects in front of
                if(closest_gt == nullptr || cmp(&it, closest_gt)) {
                    closest_gt = &it;
                }
            }
        }
    }
    if(closest_gt != nullptr) {
        result.front = closest_gt->id;
    }
    if(closest_lt != nullptr) {
        result.back = closest_lt->id;
    }

    return result;
}