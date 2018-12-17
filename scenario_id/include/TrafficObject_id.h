//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_TRAFFICOBJECT_ID_H
#define PROJECT_TRAFFICOBJECT_ID_H

#include "Lane_id.h"

class Lane_id;

class TrafficObject_id {

public:

    /**
     * compare object to compare Traffic objects by
     */
    struct Cmp {
        bool operator () (const TrafficObject_id *lhs, TrafficObject_id *rhs) {
            if(lhs->x == rhs->x)
                return lhs->id > rhs->id;
            return lhs->x < rhs->x;
        }
    };

    explicit TrafficObject_id(int id=-1, double length=0, int lane=-1, double x=0, double v=0, double a=0)
        : id(id), x(x), v(v), a(a), length(length), lane(lane) {};

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
    virtual double getAcceleration(Scenario_id &s, int leading_vehicle_id) { return 0; }

    /**
     * move a this object to a specific lane.
     * @param lane lane to move object to
     */
    void moveToLane(int lane_id);

    /**
     * remove object from any lane it may be assigned to
     */
    void removeFromLane();

    /**
     * get currently assigned lane
     * @return currently assigned lane
     */
    int getLane() const;

    double getPosition() const;
    void setPosition(double x) {
        this->x = x;
    }

    double x;

private:
    /**
     * current lane
     */
    int lane;

    void _moveToLane(Lane_id *lane);
    void _removeFromLane();

};


#endif //PROJECT_TRAFFICOBJECT_H
