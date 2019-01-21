//
// Created by maxi on 1/10/19.
//

#include "NeighborFinder.h"

Lane::NeighboringObjects NeighborFinder::getNeighboringObjects(Lane *lane, TrafficObject *trafficObject) {
    //create empty neighboringObjects struct
    auto neighboringObjects = Lane::NeighboringObjects();

    //if null reference return empty struct
    if (lane == nullptr)
        return neighboringObjects;

    //if lane is empty return empty struct
    if (lane->mTrafficObjects.size() == 0) {
       
        return neighboringObjects;
    }

    auto it = std::lower_bound(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), trafficObject, TrafficObject::Cmp());
    Lane::NeighboringObjects result;

    //auto a = &trafficObject->getLane()->road;
   

    if (it != lane->mTrafficObjects.begin())
        result.back = *(it - 1);

    if (lane->mTrafficObjects.end() == it || *it != trafficObject) {
        if (it != lane->mTrafficObjects.end())
            result.front = *it;
    } else {
        if (it + 1 != lane->mTrafficObjects.end()) {
            result.front = *(it + 1);
        }
        
    }
    return result;
}