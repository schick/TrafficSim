//
// Created by oke on 07.12.18.
//
#include <iostream>
#include <math.h>
#include <assert.h>

#include "by_id/Car.h"
#include "by_id/Road.h"
#include "by_id/Junction.h"
#include "by_id/Scenario.h"

Car::AdvanceData Car::nextStep(Scenario &s) {

    Lane::NeighboringObjects ownNeighbors = s.lanes.at(getLane()).getNeighboringObjects(s, *this);
    Road::NeighboringLanes neighboringLanes = s.roads.at(s.lanes.at(getLane()).road).getNeighboringLanes(s, s.lanes.at(getLane()));

    double m_left = getLaneChangeMetricForLane(s, neighboringLanes.left, ownNeighbors);
    double m_right = getLaneChangeMetricForLane(s, neighboringLanes.right, ownNeighbors);

    if (m_left > 1 && m_left >= m_right) {
        // go to left lane
        return Car::AdvanceData(this->id, getAcceleration(s, s.lanes.at(neighboringLanes.left).getNeighboringObjects(s, *this).front), -1);
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        return Car::AdvanceData(this->id, getAcceleration(s, s.lanes.at(neighboringLanes.right).getNeighboringObjects(s, *this).front), 1);
    }
    else {
        // stay on lane
        return Car::AdvanceData(this->id, getAcceleration(s, ownNeighbors.front), 0);
    }
}

double Car::getLaneChangeMetricForLane(Scenario &s, int neighboringLane, const Lane::NeighboringObjects &ownNeighbors) {
    if (neighboringLane != -1) {
        Lane &l = s.lanes.at(neighboringLane);
        Lane::NeighboringObjects neighbors = l.getNeighboringObjects(s, *this);
        return laneChangeMetric(s, ownNeighbors, neighbors);
    }
    return 0;
}

void Car::advanceStep(Scenario &s, Car::AdvanceData &data) {
    updateKinematicState(data);
    updateLane(s, data);
}

void Car::updateLane(Scenario &s, AdvanceData &data) {
    assert(data.car == this->id);

    // check for junction
    if (isCarOverJunction(s)) {
        moveCarAcrossJunction(s, data);
    }
    else {
        // just do a lane change if wanted
        if (data.lane_offset != 0) {
            // lane_offset should be validated in this case
            assert(s.roads.at(s.lanes.at(getLane()).road).lanes.size() > s.lanes.at(getLane()).lane_num + data.lane_offset);
            moveToLane(s.roads.at(s.lanes.at(getLane()).road).lanes[s.lanes.at(getLane()).lane_num + data.lane_offset]);
        }
    }
}

bool Car::isCarOverJunction(Scenario &s) {
    return getPosition() > s.lanes.at(getLane()).length;
}

void Car::moveCarAcrossJunction(Scenario &s, Car::AdvanceData &data) {
    assert(turns_count != 0);

    Lane &old_lane = s.lanes.at(getLane());
    Road &road = s.roads[old_lane.road];
    removeFromLane(); // important to enforce ordering of lanes!

    // subtract moved position on current lane from distance
    setPosition(getPosition() - road.length);

    // select direction based on current direction and turn
    int direction = (road.roadDir + s.turns.at(turns_begin + current_turn_offset) + 2) % 4;

    // if no road in that direction -> select next to the right
    int nextRoad;
    while ((nextRoad = s.junctions.at(road.to).outgoing[direction]) == -1) direction = (++direction) % 4;

    // move car to same or the right lane AFTER lane change
    int8_t indexOfNextLane = std::min((int8_t)s.roads.at(nextRoad).lanes.size() - 1, (int8_t)old_lane.lane_num + data.lane_offset);
    indexOfNextLane = std::max((int8_t)0, indexOfNextLane);
    while(s.roads.at(nextRoad).lanes.at(indexOfNextLane) == -1) indexOfNextLane--;
    moveToLane(s.roads.at(nextRoad).lanes.at(indexOfNextLane));

    // update next turns
    current_turn_offset = (current_turn_offset + 1) % turns_count;
}


void Car::updateKinematicState(Car::AdvanceData &data) {
    assert(data.car == id);
    a = data.acceleration;
    v = std::max(v + a, 0.);
    setPosition(getPosition() + v);
}


double Car::getAcceleration(Scenario &s, int leading_vehicle_id) {
    double vel_fraction = (v / std::min(s.roads.at(s.lanes.at(getLane()).road).limit, target_velocity));
    double without_lead = 1. - vel_fraction * vel_fraction * vel_fraction * vel_fraction; // faster than pow

    double with_lead = 0;
    if (leading_vehicle_id != -1) {
        TrafficObject &leading_vehicle = s.getTrafficObject(leading_vehicle_id);
        double delta_v = v - leading_vehicle.v;
        double s = std::max(leading_vehicle.getPosition() - getPosition() - leading_vehicle.length, min_s);
        with_lead = (min_distance + v * target_headway +
            (v * delta_v) / (2. * sqrt(max_acceleration * target_deceleration))) / s;
        with_lead = with_lead * with_lead; // faster than pow
    }
    double acceleration = max_acceleration * (without_lead - with_lead);
    return acceleration;
}

double Car::laneChangeMetric(Scenario &s, Lane::NeighboringObjects ownNeighbors, Lane::NeighboringObjects otherNeighbors) {


    if ((otherNeighbors.front == -1 || (s.getTrafficObject(otherNeighbors.front).getPosition() - getPosition()) >= (length / 2)) &&
        (otherNeighbors.back == -1 || (getPosition() - s.getTrafficObject(otherNeighbors.back).getPosition()) >= (length / 2) + min_distance)) {
        double own_wo_lc = getAcceleration(s, ownNeighbors.front);
        double own_w_lc = getAcceleration(s, otherNeighbors.front);

        double other_lane_diff = 0;
        if (otherNeighbors.back != -1) {
            other_lane_diff = (s.getTrafficObject(otherNeighbors.back).getAcceleration(s, this->id) -
                    s.getTrafficObject(otherNeighbors.back).getAcceleration(s, otherNeighbors.front));
        }


        double behind_diff = 0;
        if (ownNeighbors.back != -1) {
            behind_diff = (s.getTrafficObject(ownNeighbors.back).getAcceleration(s, ownNeighbors.front) -
                    s.getTrafficObject(ownNeighbors.back).getAcceleration(s, this->id));
        }

        if (own_w_lc > own_wo_lc) {
            return own_w_lc - own_wo_lc + politeness * (behind_diff + other_lane_diff);
        }
    }
    return 0;
}