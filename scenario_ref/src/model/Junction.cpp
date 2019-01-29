//
// Created by oke on 07.12.18.
//

#include <assert.h>

#include "model/Junction.h"
#include "model/Road.h"
#include "model/Lane.h"

Junction::Junction(uint64_t id, double x, double y)
    : id(id), x(x), y(y), incoming(), outgoing(), incoming_counter() {};

Junction::Junction(const Junction &other) :
        signals(other.signals), current_signal(other.current_signal), id(other.id), x(other.x), y(other.y),
        current_signal_time_left(other.current_signal_time_left), outgoing(other.outgoing),
        incoming(other.incoming), incoming_counter() {
    for(int i = 0; i < 4; i++) incoming_counter[i] = other.incoming_counter[i].load();
}

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
            bool isRed = !(signals.empty() || signals[current_signal].direction == i);
            for (Lane *l : incoming[i]->lanes) {
                l->trafficLight.isRed = isRed;
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