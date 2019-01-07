//
// Created by oke on 07.12.18.
//

#include "RedTrafficLight_id.h"
#include "Scenario_id.h"


RedTrafficLight_id::RedTrafficLight_id(size_t id, size_t lane, double position)
    : mAssociatedLane(lane), TrafficObject_id(id, 0, (size_t ) -1, position) {}


void RedTrafficLight_id::switchOff() {
    removeFromLane();
}

void RedTrafficLight_id::switchOn() {
    moveToLane(mAssociatedLane);
}

bool RedTrafficLight_id::isRed() {
    return getLane() != -1;
}
