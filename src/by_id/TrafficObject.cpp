//
// Created by oke on 07.12.18.
//

#include <stdexcept>
#include <assert.h>
#include <mutex>

#include "by_id/TrafficObject.h"
#include "by_id/Lane.h"
#include "by_id/Road.h"

void TrafficObject::moveToLane(int lane) {
    this->lane = lane;
}

void TrafficObject::removeFromLane() {
    this->lane = -1;
}


int TrafficObject::getLane() const {
    return lane;
}

double TrafficObject::getPosition() const {
    return x;
}
