//
// Created by oke on 07.12.18.
//

#include <stdexcept>
#include <assert.h>
#include <mutex>

#include "TrafficObject_id.h"
#include "Lane_id.h"
#include "Road_id.h"

void TrafficObject_id::moveToLane(size_t lane) {
    this->lane = lane;
}

void TrafficObject_id::removeFromLane() {
    this->lane = (size_t ) -1;
}


size_t TrafficObject_id::getLane() const {
    return lane;
}

double TrafficObject_id::getPosition() const {
    return x;
}
