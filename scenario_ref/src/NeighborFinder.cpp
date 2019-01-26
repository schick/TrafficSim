//
// Created by maxi on 1/10/19.
//

#include "NeighborFinder.h"

Lane::NeighboringObjects NeighborFinder::getNeighboringObjects(Lane *lane, Car *trafficObject) {
    //create empty neighboringObjects struct
    Lane::NeighboringObjects result;

    //if null reference or is empty return empty struct
    if (lane == nullptr || lane->mTrafficObjects.empty()) {
        return result;
    }

    auto it = std::lower_bound(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), trafficObject, TrafficObject::Cmp());

    if (it != lane->mTrafficObjects.begin()) {
        result.back = *(it - 1);
    }

    if (lane->mTrafficObjects.end() == it || *it != trafficObject) {
        if (it != lane->mTrafficObjects.end()) {
            result.front = *it;
        }
    }
    else {
        if (it + 1 != lane->mTrafficObjects.end()) {
            result.front = *(it + 1);
        }

    }
    return result;
}