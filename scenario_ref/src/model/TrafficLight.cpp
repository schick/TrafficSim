//
// Created by oke on 07.12.18.
//

#include "model/TrafficLight.h"

void TrafficLight::switchOff() {
    removeFromLane();
}

void TrafficLight::switchOn() {
    moveToLane(mAssociatedLane);
}

bool TrafficLight::isRed() {
    return getLane() != nullptr;
}
