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


    TrafficObject(uint64_t id, double length=0, double x=0, double v=0, double a=0) : id(id), x(x),v(v), a(a), length(length), lane(nullptr) {};

    /**
     * state. put acceleration in here for a more generic implementation of Car::nextStep
     */
    uint64_t id;
    double v;
    double a;
    double length;
    
    

    /**
     * get currently assigned lane
     * @return currently assigned lane
     */
    Lane *getLane();

    double getPosition();

    void setPosition(double position) { x = position; }

    //
    virtual void calcSameLaneAcceleration(TrafficObject *leadingObject){};
    virtual double getSameLaneAcceleration(){ return 0.;};


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
     * do next step
     * @param sameNeighbors
     */
    virtual void nextStep(Lane::NeighboringObjects sameNeighbors){};



private:
    /**
     * current lane
     */
    Lane *lane;


    double x;
};


#endif //PROJECT_TRAFFICOBJECT_H
