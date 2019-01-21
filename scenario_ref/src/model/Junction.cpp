//
// Created by oke on 07.12.18.
//

#include <assert.h>

#include "model/Junction.h"
#include "model/Road.h"


void Junction::initializeSignals() {
    for(int i = 0; i < 4; i++) {
        /* if initialize signals is called more than once cleanup is needed. (for example in optimization) */
        for(RedTrafficLight &tl : mRedTrafficLights[i]) tl.removeFromLane();
        mRedTrafficLights[i].clear();
        if (incoming[i] != nullptr) {
            for(Lane *l : incoming[i]->lanes)
                mRedTrafficLights[i].emplace_back(l);
        }
    }

    if(!signals.empty()) {
        current_signal = 0;
        current_signal_time_left = signals[current_signal].duration;
    }

    setSignals();
}

void Junction::setSignals() {
    for(int i = 0; i < 4; i++) {
        for(RedTrafficLight &l : mRedTrafficLights[i]) {
            if (!signals.empty() && signals[current_signal].direction == i) {
                // green light
                l.switchOff();
            } else {
                // red light
                l.switchOn();
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