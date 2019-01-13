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

    Lane(int lane, Road* road, double length) : lane(lane), road(road), mTrafficObjects(), length(length){};

    /**
     * properties
     */
    int lane;
    Road* road;
    std::mutex laneLock;
    bool isSorted = false;
    double length;

};



#endif //PROJECT_LANE_H
