//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_TRAFFICOBJECT_ID_H
#define PROJECT_TRAFFICOBJECT_ID_H

#include "vector"
#include "cuda_utils/cuda_utils.h"

class TrafficObject_id {

public:

    /**
     * state. put acceleration in here for a more generic implementation of Car::nextStep
     */
    size_t id;

    double x;
    double v;
    double a;
    double length;

    /**
     * current lane
     */
    size_t lane;
    double sameLaneAcceleration;
    double leftLaneAcceleration;
    double rightLaneAcceleration;

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
        : id(id), x(x), v(v), a(a), length(length), lane(lane), sameLaneAcceleration(0.0), leftLaneAcceleration(0.0), rightLaneAcceleration(0.0) {};


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

};

#endif //PROJECT_TRAFFICOBJECT_H
