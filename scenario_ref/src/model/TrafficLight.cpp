//
// Created by oke on 07.12.18.
//

#include "model/TrafficLight.h"

void TrafficLight::switchOff() {
    associatedLane->isRed = false;
}

void TrafficLight::switchOn() {
    associatedLane->isRed = true;
}


