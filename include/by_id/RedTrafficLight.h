//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_H
#define PROJECT_REDTRAFFICLIGHT_H

#include "by_id/TrafficObject.h"
#include "by_id/Lane.h"

class RedTrafficLight : public TrafficObject {

    int mAssociatedLane;

public:
    // traffic lights have -1 id, because traffic lights are always at the end of road.
    RedTrafficLight(int id, int lane, double position);
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
