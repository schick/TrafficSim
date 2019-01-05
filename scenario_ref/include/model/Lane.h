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

public:
    std::vector<TrafficObject *> mTrafficObjects;

    /**
     * stores neighboring objects on a lane based on a given position on lane.
     */
    struct NeighboringObjects {
        NeighboringObjects() : front(nullptr), back(nullptr) {};
        TrafficObject *front = nullptr;
        TrafficObject *back = nullptr;
    };


    Lane(int lane, Road* road) : lane(lane), road(road), mTrafficObjects() {};

    /**
     * properties
     */
    int lane;
    Road* road;
    std::mutex laneLock;

    /**
     * get current all objects on lane
     * @return list of objects on this lane
     */
    std::vector<TrafficObject*> getTrafficObjects();

    

    /**
     * get length of lane
     * @return length of lane in m
     */
    double getLength();
};



#endif //PROJECT_LANE_H
