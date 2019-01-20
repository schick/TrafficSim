//
// Created by oke on 07.12.18.
//

#include <stdexcept>
#include <assert.h>
#include <mutex>

#include "model/TrafficObject.h"
//#include "model/Lane.h"
//#include "model/Road.h"


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
    lane->isSorted = false;
}

void TrafficObject::removeFromLane() {

    if (lane == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(lane->laneLock);
    auto position = std::find(lane->mTrafficObjects.rbegin(), lane->mTrafficObjects.rend(), this);
    //auto position = std::find(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), this);
    //need -- and base() to cast from reverse pointer to forward pointer
    //lane->mTrafficObjects.erase(--position.base());
   // auto end = lane->mTrafficObjects.end() - 1;
    //auto found = --position.base();
    std::iter_swap(position, lane->mTrafficObjects.end() - 1);
    lane->mTrafficObjects.pop_back();
    lane->isSorted = false;
    lane = nullptr;
}

