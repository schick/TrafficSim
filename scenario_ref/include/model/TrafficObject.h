//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_TRAFFICOBJECT_H
#define PROJECT_TRAFFICOBJECT_H

#include "Lane.h"

class TrafficObject {

public:

    explicit TrafficObject(uint64_t id, double length = 0, double x = 0, double v = 0, double a = 0) :
        id(id), x(x), v(v), a(a), length(length) {};

    uint64_t id;

    double v;
    double a;
    double x;

    double length;

};

#endif //PROJECT_TRAFFICOBJECT_H
