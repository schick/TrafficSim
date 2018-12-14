//
// Created by oke on 07.12.18.
//

#include "TrafficObject.h"
#include "Lane.h"
#include "Road.h"
#include <stdexcept>


void Lane::prepareLanes() {
    std::sort(mTrafficObjects.begin(), mTrafficObjects.end(), TrafficObject::PosCmp());
}

std::vector<TrafficObject*> Lane::getTrafficObjects() {
    throw std::invalid_argument("Method not yet implemented");
}

Lane::NeighboringObjects Lane::getNeighboringObjects(TrafficObject *trafficObject) {
    // TODO: assert is sorted
    NeighboringObjects result;
    for(TrafficObject *to : mTrafficObjects) {
        if (to == trafficObject) {
            continue;
        }
        if (to->x > trafficObject->x) {
            result.front = to;
            return result;
        }
        if (to->x < trafficObject->x) {
            result.back = to;
        }
        // if traffic objects are on same position the object with smaller id becomes front car
        if (to->x == trafficObject->x) {
            if (trafficObject->id < to->id) {
                result.back = to;
                return result;
            }
            result.front = to;
            return result;
        }
    }
    return result;
}


double Lane::getLength() {
    return road->getLength();
}