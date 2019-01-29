//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_ID_H
#define PROJECT_REDTRAFFICLIGHT_ID_H

#include "TrafficObject_id.h"

class RedTrafficLight_id : public TrafficObject_id {

public:

    size_t mAssociatedLane;

    // traffic lights have -1 id, because traffic lights are always at the end of road.
    RedTrafficLight_id(size_t id, size_t lane, double position)
        : mAssociatedLane(lane), TrafficObject_id(id, 0, (size_t ) -1, position) {}
    /**
     * switch this light off.
     */
    CUDA_HOSTDEV void switchOff() {
        lane = (size_t) -1;
    }

    /**
     * switch this light on.
     */
    CUDA_HOSTDEV void switchOn() {
        lane = mAssociatedLane;
    }

    /**
     * @return whether red light is currently active
     */
    CUDA_HOSTDEV bool isRed() {
        return lane != (size_t ) -1;
    }

};

#endif //PROJECT_REDTRAFFICLIGHT_H
