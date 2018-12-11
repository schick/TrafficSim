//
// Created by oke on 07.12.18.
//

#include <stdexcept>
#include <assert.h>

#include "TrafficObject.h"
#include "Lane.h"


void TrafficObject::moveToLane(Lane* lane) {
    if (this->lane != nullptr) {
        removeFromLane();
    }
    assert(this->lane == nullptr);
    this->lane = lane;
    lane->mTrafficObjects.push_back(this);
}


void TrafficObject::removeFromLane() {
    if (lane == nullptr) return;
    lane->mTrafficObjects.erase(
            std::remove(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), this),
            lane->mTrafficObjects.end());
    lane = nullptr;
}


Lane *TrafficObject::getLane() {
    return lane;
}