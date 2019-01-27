//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_TRAFFICOBJECT_H
#define PROJECT_TRAFFICOBJECT_H

#include "Lane.h"

class TrafficObject {

public:

    /**
     * compare object to compare Traffic objects by
     */
    struct Cmp {
        bool operator () (const TrafficObject *lhs, TrafficObject *rhs) {
            if (lhs->position == rhs->position)
                return lhs->id > rhs->id;
            return lhs->position < rhs->position;
        }
    };

    TrafficObject(uint64_t id, double length = 0, double position = 0, double v = 0, double a = 0) : 
        id(id), position(position), v(v), a(a), length(length) {};

    uint64_t id;
    double v;
    double a;
    double length;

    double getPosition();

    virtual void setPosition(double position) { this->position = position; }

    virtual void calcSameLaneAcceleration(TrafficObject *leadingObject) {};
    virtual double getSameLaneAcceleration() { return 0.; };

private:
    double position;
};


#endif //PROJECT_TRAFFICOBJECT_H
