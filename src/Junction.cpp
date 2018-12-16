//
// Created by oke on 07.12.18.
//

#include <assert.h>

#include "Junction.h"
#include "Road.h"


void Junction::initializeSignals() {
    for(int i = 0; i < 4; i++) {
        if (incoming[i] != nullptr) {
            for(Lane *l : incoming[i]->lanes)
                mRedTrafficLights[i].emplace_back(l);
        }
    }

    if(!signals.empty()) {
        current_signal_id = 0;
        current_signal_time_left = signals[current_signal_id].duration;
    }

    setSignals();
}

void Junction::setSignals() {
    for(int i = 0; i < 4; i++) {
        for(RedTrafficLight &l : mRedTrafficLights[i]) {
            if (!signals.empty() && signals[current_signal_id].direction == i) {
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
        current_signal_id = ++current_signal_id % signals.size();
        current_signal_time_left = signals[current_signal_id].duration;
        setSignals();
    }
}