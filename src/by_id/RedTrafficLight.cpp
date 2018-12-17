//
// Created by oke on 07.12.18.
//

#include "by_id/RedTrafficLight.h"
#include "by_id/Scenario.h"


RedTrafficLight::RedTrafficLight(int id, int lane, double position)
    : mAssociatedLane(lane), TrafficObject(id, 0, -1, position) {}


void RedTrafficLight::switchOff() {
    removeFromLane();
}

void RedTrafficLight::switchOn() {
    moveToLane(mAssociatedLane);
}

bool RedTrafficLight::isRed() {
    return getLane() != -1;
}
