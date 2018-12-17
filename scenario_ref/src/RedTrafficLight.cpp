//
// Created by oke on 07.12.18.
//

#include "RedTrafficLight.h"

void RedTrafficLight::switchOff() {
    removeFromLane();
}

void RedTrafficLight::switchOn() {
    moveToLane(mAssociatedLane);
}

bool RedTrafficLight::isRed() {
    return getLane() != nullptr;
}
