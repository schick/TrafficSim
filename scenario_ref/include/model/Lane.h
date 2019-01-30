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

#include "TrafficLight.h"
#include "Road.h"

class Car;

class Lane {

public:
    void sortCars();

    std::vector<Car *> const &getCars() const {
        return mCars;
    }

    Road::NeighboringLanes neighboringLanes;

    /**
     * stores neighboring objects on a lane based on a given position on lane.
     */
    struct NeighboringObjects {
        NeighboringObjects() : front(nullptr), back(nullptr) {};
        TrafficObject *front = nullptr;
        Car *back = nullptr;
        bool operator==(const NeighboringObjects &other) {
            return other.front == front && other.back == back;
        }
    };

    Lane(int lane, Road &road, double length) : lane(lane), road(road), mCars(), length(length), trafficLight(this) {}

    /**
     * Copy Constructor
     */
    Lane(const Lane &source): road(source.road), lane(0), length(0), trafficLight(source.trafficLight) {}

    Lane::NeighboringObjects getNeighboringObjects(Car *trafficObject);

    /**
     * properties
     */
    int lane;
    Road &road;
    double length;

    TrafficLight trafficLight;

private:

    friend class Car;

    std::mutex laneLock;
    std::vector<Car *> mCars;
    bool isSorted = false;

};



#endif //PROJECT_LANE_H
