//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_TRAFFICOBJECT_H
#define PROJECT_TRAFFICOBJECT_H

#include "Lane.h"

class Lane;

class TrafficObject {

public:

    /**
     * compare object to compare Traffic objects by
     */
    struct Cmp {
        bool operator () (const TrafficObject *lhs, TrafficObject *rhs) {
            if(lhs->x == rhs->x)
                return lhs->id > rhs->id;
            return lhs->x < rhs->x;
        }
    };


    TrafficObject(int id, double length=0, double x=0, double v=0, double a=0) : id(id), x(x),v(v), a(a), length(length), lane(nullptr) {};

    /**
     * state. put acceleration in here for a more generic implementation of Car::nextStep
     */
    int id;
    double v;
    double a;
    double length;

    /**
    * calculate the acceleration with 'leading_vehicle' as lead
    * it will be assumed that 'leading_vehicle' is on current lane
    * @param leading_vehicle the leading vehicle
    * @return acceleration for t + 1
    */
    virtual double getAcceleration(TrafficObject *leading_vehicle) { return 0; }

    /**
     * move a this object to a specific lane.
     * @param lane lane to move object to
     */
    void moveToLane(Lane *lane);

    /**
     * remove object from any lane it may be assigned to
     */
    void removeFromLane();

    /**
     * get currently assigned lane
     * @return currently assigned lane
     */
    Lane *getLane();

    double getPosition();
    void setPosition(double x);

private:
    /**
     * current lane
     */
    Lane *lane;


    double x;


    void _moveToLane(Lane *lane);
    void _removeFromLane();

};


#endif //PROJECT_TRAFFICOBJECT_H