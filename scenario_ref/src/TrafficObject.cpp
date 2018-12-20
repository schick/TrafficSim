//
// Created by oke on 07.12.18.
//

#include <stdexcept>
#include <assert.h>
#include <mutex>

#include "TrafficObject.h"
#include "Lane.h"
#include "Road.h"


Lane *TrafficObject::getLane() {
    return lane;
}

double TrafficObject::getPosition() {
    return x;
}

void TrafficObject::moveToLane(Lane *lane) {
    auto a = lane->getTrafficObjects();
    this->lane = lane;
    lane->mTrafficObjects.push_back(this);
}

void TrafficObject::removeFromLane() {
    if (getLane() == nullptr) {
        return;
    }
    auto trafficObjects = getLane()->getTrafficObjects();
    trafficObjects.erase(std::remove(trafficObjects.begin(), trafficObjects.end(), this), trafficObjects.end());
}

