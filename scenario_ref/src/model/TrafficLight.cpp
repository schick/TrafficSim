//
// Created by oke on 07.12.18.
//

#include "model/TrafficLight.h"

void TrafficLight::switchOff() {
    associatedLane->isRed = false;
    //removeFromLane();
}

void TrafficLight::switchOn() {
    //moveToLane(mAssociatedLane);
    associatedLane->isRed = true;
}

bool TrafficLight::isRed() {
    return getLane() != nullptr;
}
