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
    if (lane != nullptr) {
        removeFromLane();
    }
    this->lane = lane;
    std::lock_guard<std::mutex> lock(lane->laneLock);
    lane->mTrafficObjects.push_back(this);
}

void TrafficObject::removeFromLane() {
    
    if (lane == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(lane->laneLock);
    lane ->mTrafficObjects.erase(std::remove(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), this), lane->mTrafficObjects.end());
    lane = nullptr;
}

