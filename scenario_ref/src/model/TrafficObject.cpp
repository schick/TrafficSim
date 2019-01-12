//
// Created by oke on 07.12.18.
//

#include <stdexcept>
#include <assert.h>
#include <mutex>

#include "model/TrafficObject.h"
#include "model/Lane.h"
#include "model/Road.h"


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
    auto position = std::find(lane->mTrafficObjects.rbegin(), lane->mTrafficObjects.rend(), this);
    //need -- and base() to cast from reverse pointer to forward pointer
    lane->mTrafficObjects.erase(--position.base());
    lane = nullptr;
}

