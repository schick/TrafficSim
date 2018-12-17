//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_ID_H
#define PROJECT_REDTRAFFICLIGHT_ID_H

#include "TrafficObject_id.h"
#include "Lane_id.h"

class RedTrafficLight_id : public TrafficObject_id {

    int mAssociatedLane;

public:
    // traffic lights have -1 id, because traffic lights are always at the end of road.
    RedTrafficLight_id(int id, int lane, double position);
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
