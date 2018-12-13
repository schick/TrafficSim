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
    NeighboringObjects result;
    std::sort(mTrafficObjects.begin(), mTrafficObjects.end(), TrafficObject::PosCmp());
    //ampel hat int max
    for(TrafficObject *to : mTrafficObjects) {
        if (to->x > trafficObject->x) {
            result.front = to;
            return result;
        }
        if (to->x < trafficObject->x) {
            result.back = to;
        }
    }
    return result;
}


double Lane::getLength() {
    return road->getLength();
}