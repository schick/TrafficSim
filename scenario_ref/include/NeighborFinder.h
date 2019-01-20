//
// Created by maxi on 1/6/19.
//

#ifndef TRAFFIC_SIM_NEIGHBORFINDER_H
#define TRAFFIC_SIM_NEIGHBORFINDER_H

#include <model/Lane.h>
#include <model/TrafficObject.h>
#include <model/TrafficLight.h>

class NeighborFinder {

public:
    // general, works for any lane
    static Lane::NeighboringObjects getNeighboringObjects(Lane *lane, TrafficObject *trafficObject);

//Broken!!!
//    // works only if trafficObject->lane != lane
//    static Lane::NeighboringObjects getNeighboringObjectsOnOtherLane(Lane *lane, TrafficObject *trafficObject) {
//        //create empty neighboringObjects struct
//        auto neighboringObjects = Lane::NeighboringObjects();
//
//        //if null reference return empty struct
//        if (lane == nullptr)
//            return neighboringObjects;
//
//        auto trafficObjects = lane->mTrafficObjects;
//        //if lane is empty return empty struct
//        if (trafficObjects.size() == 0)
//            return Lane::NeighboringObjects();
//
//        auto it = std::lower_bound(trafficObjects.begin(), trafficObjects.end(), trafficObject, TrafficObject::Cmp());
//        Lane::NeighboringObjects result;
//    }

};



#endif //TRAFFIC_SIM_NEIGHBORFINDER_H



