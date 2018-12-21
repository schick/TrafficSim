//
// Created by oke on 07.12.18.
//
#include <iostream>
#include <math.h>
#include <assert.h>

#include "Car_id.h"
#include "Road_id.h"
#include "Junction_id.h"
#include "Scenario_id.h"


Car_id::AdvanceData Car_id::nextStep(Scenario_id &s, size_t own_front, size_t own_back,
        size_t left_front, size_t left_back,
        size_t right_front, size_t right_back) {
    Road_id::NeighboringLanes neighboringLanes = s.roads[s.lanes[lane].road].getNeighboringLanes(s, s.lanes[lane]);

    double m_left = (neighboringLanes.left != (size_t ) -1) ? laneChangeMetric(s, left_front, left_back, own_front, own_back) : 0;
    double m_right = (neighboringLanes.right != (size_t ) -1) ? laneChangeMetric(s, right_front, right_back, own_front, own_back) : 0;

    if (m_left > 1 && m_left >= m_right) {
        // go to left lane
        return Car_id::AdvanceData(this->id, getAcceleration(s, left_front), -1);
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        return Car_id::AdvanceData(this->id, getAcceleration(s, right_front), 1);
    }
    else {
        // stay on lane
        return Car_id::AdvanceData(this->id, getAcceleration(s, own_front), 0);
    }
}


void Car_id::advanceStep(Scenario_id &s, Car_id::AdvanceData &data) {
    updateKinematicState(data);
    updateLane(s, data);
}

void Car_id::updateLane(Scenario_id &s, AdvanceData &data) {
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

bool Car_id::isCarOverJunction(Scenario_id &s) {
    return getPosition() > s.lanes.at(getLane()).length;
}

void Car_id::moveCarAcrossJunction(Scenario_id &s, Car_id::AdvanceData &data) {
    assert(turns_count != 0);

    Lane_id &old_lane = s.lanes.at(getLane());
    Road_id &road = s.roads[old_lane.road];
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


void Car_id::updateKinematicState(Car_id::AdvanceData &data) {
    assert(data.car == id);
    a = data.acceleration;
    v = std::max(v + a, 0.);
    setPosition(getPosition() + v);
}


double Car_id::getAcceleration(Scenario_id &s, size_t leading_vehicle_id) {
    double vel_fraction = (v / std::min(s.roads.at(s.lanes.at(getLane()).road).limit, target_velocity));
    double without_lead = 1. - vel_fraction * vel_fraction * vel_fraction * vel_fraction; // faster than pow

    double with_lead = 0;
    if (leading_vehicle_id != -1) {
        TrafficObject_id &leading_vehicle = s.getTrafficObject(leading_vehicle_id);
        double delta_v = v - leading_vehicle.v;
        double s = std::max(leading_vehicle.getPosition() - getPosition() - leading_vehicle.length, min_s);
        with_lead = (min_distance + v * target_headway +
            (v * delta_v) / (2. * sqrt(max_acceleration * target_deceleration))) / s;
        with_lead = with_lead * with_lead; // faster than pow
    }
    double acceleration = max_acceleration * (without_lead - with_lead);
    return acceleration;
}

double Car_id::laneChangeMetric(Scenario_id &s, size_t other_front, size_t other_back, size_t own_front, size_t own_back) {
    if ((other_front == -1 || (s.getTrafficObject(other_front).getPosition() - getPosition()) >= (length / 2)) &&
        (other_back == -1 || (getPosition() - s.getTrafficObject(other_back).getPosition()) >= (length / 2) + min_distance)) { // TODO min distance auch für vorderes auto?
        double own_wo_lc = getAcceleration(s, own_front);
        double own_w_lc = getAcceleration(s, other_front);

        double other_lane_diff = 0;
        if (other_back != -1) {
            other_lane_diff = (s.getTrafficObject(other_back).getAcceleration(s, this->id) -
                    s.getTrafficObject(other_back).getAcceleration(s, other_front));
        }


        double behind_diff = 0;
        if (own_back != -1) {
            behind_diff = (s.getTrafficObject(own_back).getAcceleration(s, own_front) -
                    s.getTrafficObject(own_back).getAcceleration(s, this->id));
        }

        if (own_w_lc > own_wo_lc) {
            return own_w_lc - own_wo_lc + politeness * (behind_diff + other_lane_diff);
        }
    }
    return 0;
}