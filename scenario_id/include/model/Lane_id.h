//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_LANE_ID_H
#define PROJECT_LANE_ID_H

#include "model/TrafficObject_id.h"
#include "cuda_utils/cuda_utils.h"

class Lane_id {

public:

    /**
     * properties
     */
    size_t id;

    size_t road;
    uint8_t lane_num;

    double length;

    size_t traffic_light;

    /**
     * stores neighboring objects on a lane based on a given position on lane.
     */
    struct NeighboringObjects {
        CUDA_HOSTDEV NeighboringObjects() : front((size_t)-1), back((size_t)-1) {};
        size_t front = (size_t)-1;
        size_t back = (size_t)-1;
    };

    struct NeighboringObjectsRef {
        CUDA_HOSTDEV NeighboringObjectsRef() : front(nullptr), back(nullptr) {};
        CUDA_HOSTDEV NeighboringObjectsRef(TrafficObject_id *front, TrafficObject_id *back)
            : front(front), back(back) {};
        TrafficObject_id *front = nullptr;
        TrafficObject_id *back = nullptr;
    };

    Lane_id(uint8_t lane_num, size_t road, size_t id, double length)
            : lane_num(lane_num), road(road), id(id), length(length), traffic_light((size_t ) -1) {};

};

#endif //PROJECT_LANE_H
