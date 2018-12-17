//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_TRAFFICOBJECT_H
#define PROJECT_TRAFFICOBJECT_H

#include "by_id/Lane.h"

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

    explicit TrafficObject(int id=-1, double length=0, int lane=-1, double x=0, double v=0, double a=0)
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
    virtual double getAcceleration(Scenario &s, int leading_vehicle_id) { return 0; }

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

    void _moveToLane(Lane *lane);
    void _removeFromLane();

};


#endif //PROJECT_TRAFFICOBJECT_H
