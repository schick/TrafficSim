//
// Created by maxi on 1/6/19.
//

#ifndef TRAFFIC_SIM_NEIGHBORFINDER_H
#define TRAFFIC_SIM_NEIGHBORFINDER_H

#include <model/Lane.h>
#include <model/TrafficObject.h>
#include <model/TrafficLight.h>
#include <model/Car.h>

class NeighborFinder {

public:
    // general, works for any lane
    static Lane::NeighboringObjects getNeighboringObjects(Lane *lane, Car *trafficObject);

};



#endif //TRAFFIC_SIM_NEIGHBORFINDER_H



