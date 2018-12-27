//
// Created by oke on 07.12.18.
//

#include "TrafficObject.h"
#include "Lane.h"
#include "Road.h"
#include <stdexcept>

std::vector<TrafficObject*> Lane::getTrafficObjects() {
    return mTrafficObjects;
}


double Lane::getLength() {
    return road->getLength();
}