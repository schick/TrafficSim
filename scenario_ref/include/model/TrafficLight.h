//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_H
#define PROJECT_REDTRAFFICLIGHT_H

#include "TrafficObject.h"
#include <stdexcept>

class TrafficLight : public TrafficObject {

    Lane* associatedLane;

public:
    // traffic lights have -1 id, because traffic lights are always at the end of road.
    explicit TrafficLight(Lane *lane) : associatedLane(lane), TrafficObject(-1, 0, lane->length - 35. / 2.) {}
    /**
     * switch this light off.
     */
    void switchOff();

    /**
     * switch this light on.
     */
    void switchOn();

};

#endif //PROJECT_REDTRAFFICLIGHT_H
