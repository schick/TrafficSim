//
// Created by oke on 07.12.18.
//

#include "TrafficObject.h"
#include "Lane.h"
#include "Road.h"
#include <stdexcept>

std::vector<TrafficObject*> Lane::getTrafficObjects() {
    throw std::invalid_argument("Method not yet implemented");
}

Lane::NeighboringObjects Lane::getNeighboringObjects(TrafficObject *trafficObject) {
    // TODO: assert is sorted
    if (mTrafficObjects.size() == 0) return NeighboringObjects();
    auto it = std::lower_bound(mTrafficObjects.begin(), mTrafficObjects.end(), trafficObject, TrafficObject::Cmp());
    NeighboringObjects result;

    if (it != mTrafficObjects.begin())
        result.back = *(it - 1);

    if (mTrafficObjects.end() == it || *it != trafficObject) {
        if (it != mTrafficObjects.end())
            result.front = *it;
    } else {
        if (it + 1 != mTrafficObjects.end())
            result.front = *(it + 1);
    }
    return result;
}


double Lane::getLength() {
    return road->getLength();
}