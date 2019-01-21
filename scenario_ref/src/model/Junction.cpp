//
// Created by oke on 07.12.18.
//

#include <assert.h>

#include "model/Junction.h"
#include "model/Road.h"


void Junction::initializeSignals() {

    if(!signals.empty()) {
        current_signal = 0;
        current_signal_time_left = signals[current_signal].duration;
    }

    setSignals();
}

void Junction::setSignals() {

    for (int i = 0; i < 4; i++) {
        if (incoming[i] != nullptr) {
            for (Lane *l : incoming[i]->lanes) {
                if (!signals.empty() &&
                    signals[current_signal].direction == i) {
                    // green light
                    l->isRed = false;
                }
                else {
                    // red light
                    l->isRed = true;
                }
            }

        }
    }

}

void Junction::updateSignals() {
    if(!signals.empty() && 0 == --current_signal_time_left) {
        current_signal = ++current_signal % signals.size();
        current_signal_time_left = signals[current_signal].duration;
        setSignals();
    }
}

std::vector<Junction::Direction> Junction::getPossibleDirections() {
    std::vector<Junction::Direction> directionVector;
    for (int i = 0; i < 4; i++) {
        if (incoming[i] != nullptr) {
            directionVector.emplace_back((Direction) i);
        }
    }

    return directionVector;
}