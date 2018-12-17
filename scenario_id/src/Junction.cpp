//
// Created by oke on 07.12.18.
//

#include <assert.h>

#include "Scenario_id.h"
#include "Junction_id.h"
#include "Road_id.h"

void Junction_id::initializeSignals(Scenario_id &s) {
    if(signal_count != 0) {
        current_signal_id = 0;
        current_signal_time_left = s.signals.at(signal_begin + current_signal_id).duration;
    }
    setSignals(s);
}

void Junction_id::setSignals(Scenario_id &s) {
    if (signal_count == 0)
        return;
    for(int i = 0; i < 4; i++) {
        for(int light_id : red_traffic_lights_ids[i]) {
            if(light_id != -1) {
                RedTrafficLight_id &l = s.traffic_lights[light_id - s.cars.size()];
                if (s.signals.at(signal_begin + current_signal_id).direction == i) {
                    // green light
                    l.switchOff();
                } else {
                    // red light
                    l.switchOn();
                }
            }
        }
    }
}

void Junction_id::updateSignals(Scenario_id &s) {
    if(signal_count != 0 && 0 == --current_signal_time_left) {
        current_signal_id = ++current_signal_id % signal_count;
        current_signal_time_left = s.signals.at(signal_begin + current_signal_id).duration;
        setSignals(s);
    }
}