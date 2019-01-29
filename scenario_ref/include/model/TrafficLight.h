//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_REDTRAFFICLIGHT_H
#define PROJECT_REDTRAFFICLIGHT_H

#include "TrafficObject.h"

class Lane;

class TrafficLight : public TrafficObject {

public:

    explicit TrafficLight(Lane *lane);

    bool isRed;
};

#endif //PROJECT_REDTRAFFICLIGHT_H
