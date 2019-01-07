//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_LANE_ID_H
#define PROJECT_LANE_ID_H

#include <inttypes.h>
#include <vector>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

class Scenario_id;
class TrafficObject_id;

class Lane_id {

public:

    /**
     * properties
     */
    uint8_t lane_num;
    size_t id;
    size_t road;
    double length;

    /**
     * stores neighboring objects on a lane based on a given position on lane.
     */
    struct NeighboringObjects {
        CUDA_HOSTDEV NeighboringObjects() : front((size_t)-1), back((size_t)-1) {};
        CUDA_HOSTDEV NeighboringObjects(size_t front, size_t back) : front(front), back(back) {};
        size_t front = (size_t)-1;
        size_t back = (size_t)-1;
    };

    Lane_id(uint8_t lane_num, size_t road, size_t id, double length) : lane_num(lane_num), road(road), id(id), length(length) {};

    /**
     * get neighboring objects on this lane
     * @param object find neigbhoring objects for this object. may actually be on a different lane.
     *      this algorithm treats all objects as if there where on this lane.
     * @return neighboring objects
     */
    NeighboringObjects getNeighboringObjects(const Scenario_id &s, const TrafficObject_id &object) const;

};



#endif //PROJECT_LANE_H
