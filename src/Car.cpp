//
// Created by oke on 07.12.18.
//

#include "Car.h"
#include "Road.h"
#include <iostream>
#include <math.h>
#include <assert.h>

Car::AdvanceData Car::nextStep() {
    /**
     * lots of cases...
     */
    Lane::NeighboringObjects ownNeighbors = getLane()->getNeighboringObjects(this);
    std::vector<Lane*> neighboringLanes = getLane()->road->getNeighboringLanes(getLane());

    std::vector<double> m;
    for(Lane *lane : neighboringLanes) {
        Lane::NeighboringObjects leftNeighbors = lane->getNeighboringObjects(this);
        m.emplace_back(laneChangeMetic(ownNeighbors, leftNeighbors));
    }

    if (m.size() == 1 && m[0] > 1) {
            return Car::AdvanceData(this, getAcceleration(neighboringLanes[0]->getNeighboringObjects(this).front), neighboringLanes[0]);
    } else if (m.size() == 2){
        if (m[0] > 1 && m[0] >= m[1]) {
            return Car::AdvanceData(this, getAcceleration(neighboringLanes[0]->getNeighboringObjects(this).front), neighboringLanes[0]);
        } else if (m[1] > 1) {
            return Car::AdvanceData(this, getAcceleration(neighboringLanes[1]->getNeighboringObjects(this).front), neighboringLanes[1]);
        }
    }
    return Car::AdvanceData(this, getAcceleration(ownNeighbors.front), nullptr);
}

void Car::advanceStep(AdvanceData data) {
    assert(data.car == this);
    a = data.acceleration;
    v = ((v + a) < 0) ? 0 : v + a;
    x = x + v;
    // TODO: junctions
    if (data.lane_change)
        moveToLane(data.lane_change);
}


double Car::getAcceleration(TrafficObject *leading_vehicle) {
    double vel_fraction = (v / target_velocity);
    double without_lead = 1. - vel_fraction * vel_fraction * vel_fraction * vel_fraction;

    double with_lead = 0;
    if (leading_vehicle != nullptr) {
        double delta_v = v - leading_vehicle->v;
        double s = leading_vehicle->x - x - leading_vehicle->length;
        with_lead = (min_distance + v * target_headway +
                     (v * delta_v) / (2 * sqrt(max_acceleration * target_deceleration))) / s;
        with_lead = with_lead * with_lead;
    }
    double acceleration = max_acceleration * (without_lead - with_lead);
    return acceleration;
}

double Car::laneChangeMetic(Lane::NeighboringObjects ownNeighbors, Lane::NeighboringObjects otherNeighbors) {

    if ((otherNeighbors.front == nullptr || (otherNeighbors.front->x - x) >= (length / 2)) &&
        (otherNeighbors.back == nullptr || (x - otherNeighbors.back->x) >= (length / 2) + min_distance)) {
        double own_wo_lc = getAcceleration(ownNeighbors.front);
        double own_w_lc = getAcceleration(otherNeighbors.front);

        double other_lane_diff = 0;
        if (otherNeighbors.back != nullptr)
            other_lane_diff = (otherNeighbors.back->getAcceleration(this) -
                    otherNeighbors.back->getAcceleration(otherNeighbors.front));

        double behind_diff = 0;
        if (ownNeighbors.back != nullptr)
            behind_diff = (ownNeighbors.back->getAcceleration(ownNeighbors.front) -
                    ownNeighbors.back->getAcceleration(this));

        if (own_w_lc > own_wo_lc) {
            return own_w_lc - own_wo_lc + politeness * (behind_diff + other_lane_diff);
        }
    }
    return 0;
}