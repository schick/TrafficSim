//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_ID_H
#define PROJECT_REDTRAFFICLIGHT_ID_H

#include "TrafficObject_id.h"
#include "Lane_id.h"

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
#endif


class RedTrafficLight_id : public TrafficObject_id {

public:

    size_t mAssociatedLane;

    // traffic lights have -1 id, because traffic lights are always at the end of road.
    RedTrafficLight_id(size_t id, size_t lane, double position);
    /**
     * switch this light off.
     */
    CUDA_HOSTDEV void switchOff();

    /**
     * switch this light on.
     */
    CUDA_HOSTDEV void switchOn();

    /**
     * @return whether red light is currently active
     */
    CUDA_HOSTDEV bool isRed() {
        return lane != -1;
    }

};

#endif //PROJECT_REDTRAFFICLIGHT_H
