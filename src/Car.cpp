//
// Created by oke on 07.12.18.
//

#include "Car.h"
#include "Road.h"
#include <iostream>
#include <math.h>
#include <assert.h>
#include <Junction.h>

Car::AdvanceData Car::nextStep() {
    /**
     * lots of cases...
     */
    Lane::NeighboringObjects ownNeighbors = getLane()->getNeighboringObjects(this);
    Road::NeighboringLanes neighboringLanes = getLane()->road->getNeighboringLanes(getLane());

    double m_left = 0;
    double m_right = 0;

    if (neighboringLanes.left != nullptr) {
        Lane::NeighboringObjects leftNeighbors = neighboringLanes.left->getNeighboringObjects(this);
        m_left = laneChangeMetric(ownNeighbors, leftNeighbors);
    }

    if (neighboringLanes.right != nullptr) {
        Lane::NeighboringObjects rightNeighbors = neighboringLanes.right->getNeighboringObjects(this);
        m_right = laneChangeMetric(ownNeighbors, rightNeighbors);
    }

    if(m_left > 1 && m_left >= m_right) {
        // left
        return Car::AdvanceData(this, getAcceleration(neighboringLanes.left->getNeighboringObjects(this).front), -1);
    } else if(m_right > 1 && m_left < m_right) {
        // right
        return Car::AdvanceData(this, getAcceleration(neighboringLanes.right->getNeighboringObjects(this).front), 1);
    } else {
        // stay
        return Car::AdvanceData(this, getAcceleration(ownNeighbors.front), 0);
    }
}

void Car::advanceStep(AdvanceData data) {
    assert(data.car == this);
    // update kinematic state
    a = data.acceleration;
    v = std::max(v + a, 0.);
    x = x + v;

    // check for junction
    if (x > getLane()->getLength()) {
        assert(!turns.empty());

        // subtract moved position on current lane from distance
        x -= getLane()->road->getLength();

        // select direction based on current turn
        int direction = (getLane()->road->getDirection() + turns.front()) % 4;

        // if no road in that direction -> select next to the right
        Road *road;
        while((road = getLane()->road->to->outgoing[direction]) == nullptr) direction = (++direction) % 4;

        // move car to same or the right lane AFTER lane change
        int16_t lane_id = std::max(0, std::min((int) road->lanes.size() - 1, getLane()->lane_id + data.lane_offset));
        moveToLane(road->lanes[lane_id]);

        // update next turns
        turns.push_back(turns.front());
        turns.pop_front();

    } else {
        // just do a lane change if wanted
        if (data.lane_offset != 0)
            // lane_offset should be validated in this case
            moveToLane(getLane()->road->lanes[getLane()->lane_id + data.lane_offset]);
    }
}


double Car::getAcceleration(TrafficObject *leading_vehicle) {
    double vel_fraction = (v / std::min(getLane()->road->limit, target_velocity));
    double without_lead = 1. - std::pow(vel_fraction, 4);

    double with_lead = 0;
    if (leading_vehicle != nullptr) {
        double delta_v = v - leading_vehicle->v;
        double s = leading_vehicle->x - x - leading_vehicle->length;
        with_lead = (min_distance + v * target_headway +
                     (v * delta_v) / (2 * sqrt(max_acceleration * target_deceleration))) / s;
        with_lead = std::pow(with_lead, 2);
    }
    double acceleration = max_acceleration * (without_lead - with_lead);
    return acceleration;
}

double Car::laneChangeMetric(Lane::NeighboringObjects ownNeighbors, Lane::NeighboringObjects otherNeighbors) {

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