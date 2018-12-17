//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_LANE_ID_H
#define PROJECT_LANE_ID_H

#include <inttypes.h>

class Scenario_id;
class TrafficObject_id;

class Lane_id {

public:

    /**
     * properties
     */
    uint8_t lane_num;
    int id;
    int road;
    double length;

    /**
     * stores neighboring objects on a lane based on a given position on lane.
     */
    struct NeighboringObjects {
        NeighboringObjects() : front(-1), back(-1) {};
        int front = -1;
        int back = -1;
    };

    Lane_id(uint8_t lane_num, int road, int id, double length) : lane_num(lane_num), road(road), id(id), length(length) {};

    /**
     * get neighboring objects on this lane
     * @param object find neigbhoring objects for this object. may actually be on a different lane.
     *      this algorithm treats all objects as if there where on this lane.
     * @return neighboring objects
     */
    NeighboringObjects getNeighboringObjects(Scenario_id &s, TrafficObject_id &object);

};



#endif //PROJECT_LANE_H
