//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_H
#define PROJECT_REDTRAFFICLIGHT_H

#include "TrafficObject.h"
#include "Lane.h"
#include <stdexcept>


class RedTrafficLight : public TrafficObject {

    Lane* mAssociatedLane;

public:
    // traffic lights have -1 id, because traffic lights are always at the end of road.
    explicit RedTrafficLight(Lane *lane) : mAssociatedLane(lane), TrafficObject(-1, 0, lane->getLength() - 35. / 2.) {}

    /**
     * switch this light off.
     */
    void switchOff();

    /**
     * switch this light on.
     */
    void switchOn();

    /**
     * @return whether red light is currently active
     */
    bool isRed();

};

#endif //PROJECT_REDTRAFFICLIGHT_H