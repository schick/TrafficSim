//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_LANE_H
#define PROJECT_LANE_H

#include <vector>
#include <inttypes.h>
#include <iostream>
#include <algorithm>
#include <mutex>

class Road;
class TrafficObject;

class Lane {

    friend class TrafficObject;

private:

    /**
     * object on this lane. currently not sorted.
     */
    std::vector<TrafficObject *> mTrafficObjects;
    std::mutex mTrafficObjectsMutex;

public:

    void prepareLanes();

    /**
     * stores neighboring objects on a lane based on a given position on lane.
     */
    struct NeighboringObjects {
        NeighboringObjects() : front(nullptr), back(nullptr) {};
        TrafficObject *front = nullptr;
        TrafficObject *back = nullptr;
    };


    Lane(uint8_t lane_id, Road* road) : lane_id(lane_id), road(road), mTrafficObjects(), mTrafficObjectsMutex() {};

    /**
     * properties
     */
    uint8_t lane_id;
    Road* road;

    /**
     * get current all objects on lane
     * @return list of objects on this lane
     */
    std::vector<TrafficObject*> getTrafficObjects();

    /**
     * get neighboring objects on this lane
     * @param object find neigbhoring objects for this object. may actually be on a different lane.
     *      this algorithm treats all objects as if there where on this lane.
     * @return neighboring objects
     */
    NeighboringObjects getNeighboringObjects(TrafficObject *object);

    /**
     * get length of lane
     * @return length of lane in m
     */
    double getLength();
};



#endif //PROJECT_LANE_H
