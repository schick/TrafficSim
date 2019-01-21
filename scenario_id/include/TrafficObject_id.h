//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_TRAFFICOBJECT_ID_H
#define PROJECT_TRAFFICOBJECT_ID_H

#include "Lane_id.h"
#include "vector"
#include <iostream>

class Lane_id;

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


class TrafficObject_id {


public:

    /**
     * compare object to compare Traffic objects by
     */
    struct Cmp {
        CUDA_HOSTDEV bool operator () (const TrafficObject_id *lhs, const TrafficObject_id *rhs) {
            if(lhs == nullptr) return false;
            if(rhs == nullptr) return true;
            if(lhs->x == rhs->x)
                return lhs->id > rhs->id;
            return lhs->x < rhs->x;
        }
    };

    explicit TrafficObject_id(size_t id=(size_t)-1, double length=0, size_t lane=(size_t)-1, double x=0, double v=0, double a=0)
        : id(id), x(x), v(v), a(a), length(length), lane(lane) {};

    /**
     * state. put acceleration in here for a more generic implementation of Car::nextStep
     */
    size_t id;
    double v;
    double a;
    double length;

    /**
    * calculate the acceleration with 'leading_vehicle' as lead
    * it will be assumed that 'leading_vehicle' is on current lane
    * @param leading_vehicle the leading vehicle
    * @return acceleration for t + 1
    */
    virtual double getAcceleration(Scenario_id &s, size_t leading_vehicle_id) { return 0; }

    /**
     * move a this object to a specific lane.
     * @param lane lane to move object to
     */
    void moveToLane(size_t lane_id);

    /**
     * remove object from any lane it may be assigned to
     */
    void removeFromLane();

    /**
     * get currently assigned lane
     * @return currently assigned lane
     */
    size_t getLane() const;

    double getPosition() const;
    void setPosition(double x) {
        this->x = x;
    }

    double x;


    CUDA_HOSTDEV inline bool operator<(const TrafficObject_id &r) const {
        if (lane == r.lane) {
            if(x == r.x)
                return id > r.id;
            return x < r.x;
        }
        return lane < r.lane;
    }

    CUDA_HOSTDEV inline bool operator<=(const TrafficObject_id &r) const {
        if (lane == r.lane) {
            if(x == r.x)
                return id >= r.id;
            return x <= r.x;
        }
        return lane <= r.lane;
    }

    CUDA_HOSTDEV inline bool operator>(const TrafficObject_id &r) const {
        if (lane == r.lane) {
            if(x == r.x)
                return id < r.id;
            return x > r.x;
        }
        return lane > r.lane;
    }

    CUDA_HOSTDEV inline bool operator>=(const TrafficObject_id &r) const {
        if (lane == r.lane) {
            if(x == r.x)
                return id <= r.id;
            return x >= r.x;
        }
        return lane >= r.lane;
    }


    CUDA_HOSTDEV inline bool operator==(const TrafficObject_id &r) const {
        return (lane == r.lane) && (x == r.x) && (id == r.id);
    }


    /**
     * current lane
     */
    size_t lane;


private:

    void _moveToLane(Lane_id *lane);
    void _removeFromLane();

};

#endif //PROJECT_TRAFFICOBJECT_H
