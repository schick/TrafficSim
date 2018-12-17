//
// Created by oke on 07.12.18.
//

#include "by_id/Scenario.h"

Lane::NeighboringObjects Lane::getNeighboringObjects(Scenario &s, TrafficObject &object) {
    // TODO: assert is sorted
    NeighboringObjects result;

    TrafficObject::Cmp cmp; //  a < b

    TrafficObject *closest_gt = nullptr;
    TrafficObject *closest_lt = nullptr;

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