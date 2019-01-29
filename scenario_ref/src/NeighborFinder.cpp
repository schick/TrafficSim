//
// Created by maxi on 1/10/19.
//

#include "NeighborFinder.h"

Lane::NeighboringObjects NeighborFinder::getNeighboringObjects(Lane *lane, Car *trafficObject) {
    //create empty neighboringObjects struct
    Lane::NeighboringObjects result;

    //if null reference or is empty return empty struct
    if (lane == nullptr || lane->getCars().empty()) {
        return result;
    }

    auto it = std::lower_bound(lane->getCars().begin(), lane->getCars().end(), trafficObject, Car::Cmp());

    if (it != lane->getCars().begin()) {
        result.back = *(it - 1);
    }

    if (lane->getCars().end() == it || *it != trafficObject) {
        if (it != lane->getCars().end()) {
            result.front = *it;
        }
    }
    else {
        if (it + 1 != lane->getCars().end()) {
            result.front = *(it + 1);
        }

    }
    return result;
}