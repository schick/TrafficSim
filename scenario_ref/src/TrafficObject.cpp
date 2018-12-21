//
// Created by oke on 07.12.18.
//

#include <stdexcept>
#include <assert.h>
#include <mutex>

#include "TrafficObject.h"
#include "Lane.h"
#include "Road.h"

void TrafficObject::moveToLane(Lane* lane) {
    if (lane == this->lane) return;
    if (this->lane == nullptr) {
        std::lock_guard<std::mutex> lk(lane->mTrafficObjectsMutex);
        _moveToLane(lane);
    } else {
        std::lock(lane->mTrafficObjectsMutex, this->lane->mTrafficObjectsMutex);
        std::lock_guard<std::mutex> lk1(lane->mTrafficObjectsMutex, std::adopt_lock);
        std::lock_guard<std::mutex> lk2(this->lane->mTrafficObjectsMutex, std::adopt_lock);
        _moveToLane(lane);
    }
}

void TrafficObject::removeFromLane() {
    if (lane == nullptr) return;
    std::lock_guard<std::mutex> lock(lane->mTrafficObjectsMutex);
    _removeFromLane();
}

void TrafficObject::_moveToLane(Lane* lane) {
    if (this->lane != nullptr) {
        _removeFromLane();
    }
    assert(this->lane == nullptr);
    this->lane = lane;

    auto it = std::upper_bound(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), this, TrafficObject::Cmp());
    lane->mTrafficObjects.insert(it, this);
}

void TrafficObject::_removeFromLane() {
    if (lane == nullptr) return;
    auto it = std::lower_bound(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), this, TrafficObject::Cmp());
    assert(*it == this);
    lane->mTrafficObjects.erase(it);
    lane = nullptr;
}

Lane *TrafficObject::getLane() {
    return lane;
}

double TrafficObject::getPosition() {
    return x;
}

void TrafficObject::setPosition(double x) {
    if (lane == nullptr) {
        this->x = x;
    } else {
        std::lock_guard<std::mutex> lock(lane->mTrafficObjectsMutex);
        auto it = std::lower_bound(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), this, Cmp());
        Lane *lane = this->lane;
        if (((it + 1) != lane->mTrafficObjects.end()) && (*(it + 1))->x < x) {
            _removeFromLane();
            this->x = x;
            _moveToLane(lane);
        } else {
            this->x = x;
        }
    }
}