//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_H
#define PROJECT_REDTRAFFICLIGHT_H

#include "TrafficObject.h"
#include "Lane.h"
#include <stdexcept>


class RedTrafficLight : public TrafficObject {

    RedTrafficLight(Lane *lane) : associatedLane(lane), TrafficObject(0, lane->getLength()) {}

    Lane* associatedLane;

    /**
     * switch this light off.
     */
    void switchOff() {
        throw std::invalid_argument("not implemented yet");
    }


    /**
     * switch this light on.
     */
    void switchOn() {
        throw std::invalid_argument("not implemented yet");
    }

};



#endif //PROJECT_REDTRAFFICLIGHT_H
