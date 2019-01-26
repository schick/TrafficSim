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
//#include "TrafficLight.h"
//#include "TrafficObject.h"

class Road;
class TrafficObject;
class TrafficLight;

class Lane {


public:
    std::vector<TrafficObject *> mTrafficObjects;
    //TrafficLight &trafficLight;
    bool isRed = false;

    /**
     * stores neighboring objects on a lane based on a given position on lane.
     */
    struct NeighboringObjects {
        NeighboringObjects() : front(nullptr), back(nullptr) {};
        TrafficObject *front = nullptr;
        TrafficObject *back = nullptr;
    };

    Lane(int lane, Road &road, double length) : lane(lane), road(road), mTrafficObjects(), length(length) {

    }

    /**
     * Copy Constructor
     */
    Lane(const Lane &source): road(source.road) {}

    /**
     * properties
     */
    int lane;
    Road &road;
    std::mutex laneLock;
    bool isSorted = false;
    double length;

};



#endif //PROJECT_LANE_H
