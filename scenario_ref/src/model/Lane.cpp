//
// Created by oke on 07.12.18.
//

#include "model/TrafficObject.h"
#include "model/Lane.h"
#include "model/Road.h"
#include <stdexcept>

std::vector<TrafficObject*> Lane::getTrafficObjects() {
    return mTrafficObjects;
}


double Lane::getLength() {
    return road->getLength();
}