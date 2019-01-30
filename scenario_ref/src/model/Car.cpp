//
// Created by oke on 07.12.18.
//

#include "model/Car.h"
#include "model/Road.h"
#include "model/Scenario.h"

#include <algorithm>

Car::Car(uint64_t id, double length, double target_velocity, double max_acceleration, double target_deceleration,
    double min_distance, double target_headway, double politeness, double x, double v, double a)
        : target_velocity(target_velocity), max_acceleration(max_acceleration), target_deceleration(target_deceleration),
            min_distance(min_distance), target_headway(target_headway), politeness(politeness), lane(nullptr),
            TrafficObject(id, length, x, v, a) {}

void Car::prepareNextMove() {

    Lane *lane = getLane();

    calcSameLaneAcceleration(sameNeighbors.front);

    double m_left = getLaneChangeMetric(sameNeighbors, lane->neighboringLanes.left, leftNeighbors, true);
    double m_right = getLaneChangeMetric(sameNeighbors, lane->neighboringLanes.right, rightNeighbors, false);

    if (m_left > 1 && m_left >= m_right) {
        // go to left lane
        advance_data.new_acceleration = advance_data.leftLaneAcceleration;
        advance_data.new_lane_offset = -1;
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        advance_data.new_acceleration = advance_data.rightLaneAcceleration;
        advance_data.new_lane_offset = 1;
    }
    else {
        // stay on lane
        advance_data.new_acceleration = advance_data.sameLaneAcceleration;
        advance_data.new_lane_offset = 0;
    }
}

void Car::makeNextMove(Scenario &scenario) {
    updateKinematicState();
    updateLane(scenario);
}

void Car::calcSameLaneAcceleration(TrafficObject *leadingObject) {
    advance_data.sameLaneAcceleration = getAcceleration(leadingObject);
}

void Car::updateKinematicState() {
    a = advance_data.new_acceleration;
    v = std::max(v + a, 0.);
    x += v;
    travelledDistance += v;
}

void Car::moveToLane(Lane &lane) {
    if (this->lane != nullptr) {
        removeFromLane();
    }
    this->lane = &lane;
    std::lock_guard<std::mutex> lock(lane.laneLock);
    lane.mCars.push_back(this);
    lane.isSorted = false;
}

void Car::removeFromLane() {
    if (lane == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(lane->laneLock);
    auto position = std::find(lane->mCars.rbegin(), lane->mCars.rend(), this);
    std::iter_swap(position, lane->mCars.end() - 1);
    lane->mCars.pop_back();
    lane->isSorted = false;
    lane = nullptr;
}

Lane *Car::getLane() {
    return lane;
}

void Car::updateLane(Scenario &scenario) {
    // check for junction
    if (isCarOverJunction()) {
        moveCarAcrossJunction(scenario);
    }
    else {
        // just do a lane change if wanted
        if (advance_data.new_lane_offset != 0) {
            // lane_offset should be validated in this case
            moveToLane(*lane->road.lanes[lane->lane + advance_data.new_lane_offset]);
        }
    }
}

void Car::moveCarAcrossJunction(Scenario &scenario) {
    Lane *old_lane = lane;

    // subtract moved position on current lane from distance
    auto oldLaneLength = old_lane->length;
    x -= oldLaneLength;

    auto &incoming_counter = old_lane->road.to->incoming_counter[(old_lane->road.getDirection() + 2) % 4];
    incoming_counter += (unsigned long) v / scenario.total_steps * (scenario.total_steps - scenario.current_step) * 100;
    // select direction based on current direction and turn
    int direction = (old_lane->road.getDirection() + turns.front() + 2) % 4;

    // if no road in that direction -> select next to the right
    Road *nextRoad;
    while ((nextRoad = old_lane->road.to->outgoing[direction]) == nullptr) direction = (++direction) % 4;

    // move car to same or the right lane AFTER lane change
    int indexOfNextLane = std::min((int)nextRoad->lanes.size() - 1, old_lane->lane + advance_data.new_lane_offset);
    indexOfNextLane = std::max(0, indexOfNextLane);

    moveToLane(*nextRoad->lanes[indexOfNextLane]);

    // update next turns
    turns.push_back(turns.front());
    turns.pop_front();
}

bool Car::isCarOverJunction() {
    return x >= lane->length;
}

double Car::getAcceleration(TrafficObject *leading_vehicle) {
    double vel_fraction = (v / std::min(lane->road.limit, target_velocity));
    double without_lead = 1. - vel_fraction * vel_fraction * vel_fraction * vel_fraction; // faster than pow

    // TODO: move this check to getNeighboringObjects...
    if (lane->trafficLight.isRed) {
        if (leading_vehicle == nullptr || leading_vehicle->x >= lane->trafficLight.x) {
            if (x < lane->trafficLight.x) {
                leading_vehicle = &lane->trafficLight;
            }
        }
    }

    double with_lead = 0.0;
    if (leading_vehicle != nullptr) {
        with_lead = calculateWithLead(*leading_vehicle);
    }

    double acceleration = max_acceleration * (without_lead - with_lead);
    return acceleration;
}

double Car::calculateWithLead(TrafficObject &leading_vehicle)
{
    double delta_v = v - leading_vehicle.v;
    double s = std::max(leading_vehicle.x - x - leading_vehicle.length, min_s);
    double with_lead = (min_distance + v * target_headway +
                 (v * delta_v) / (2. * sqrt(max_acceleration * target_deceleration))) / s;
    with_lead = with_lead * with_lead; // faster than pow

    return with_lead;
}

void Car::setLeadingTrafficObject(TrafficObject * &leading_vehicle, TrafficLight &trafficLight) {


}

double Car::getLaneChangeMetric(Lane::NeighboringObjects &sameNeighbors, Lane *otherLane,
        Lane::NeighboringObjects &otherNeighbors, bool isLeftLane) {
    if (otherLane == nullptr)
        return 0;
    return calculateLaneChangeMetric(sameNeighbors, otherNeighbors, isLeftLane);
}

double Car::calculateLaneChangeMetric(Lane::NeighboringObjects &sameNeighbors, Lane::NeighboringObjects &otherNeighbors, bool isLeftLane) {
    if (hasFrontSpaceOnOtherLane(otherNeighbors) && hasBackSpaceOnOtherLane(otherNeighbors)) {

        double sameLaneAcceleration = advance_data.sameLaneAcceleration;
        double otherLaneAcceleration = getAcceleration(otherNeighbors.front);

        if (otherLaneAcceleration > sameLaneAcceleration) {

            if (isLeftLane) {
                advance_data.leftLaneAcceleration = otherLaneAcceleration;
            }
            else {
                advance_data.rightLaneAcceleration = otherLaneAcceleration;
            }

            double other_lane_diff = 0;
            if (otherNeighbors.back != nullptr) {
                other_lane_diff = (otherNeighbors.back->getAcceleration(this) -
                                   otherNeighbors.back->getAcceleration(otherNeighbors.front));
            }


            double behind_diff = 0;
            if (sameNeighbors.back != nullptr) {
                behind_diff = (sameNeighbors.back->getAcceleration(sameNeighbors.front) -
                               sameNeighbors.back->getAcceleration(this));
            }

            return otherLaneAcceleration - sameLaneAcceleration + politeness * (behind_diff + other_lane_diff);
        }
    }
    return 0.0;
}

bool Car::hasFrontSpaceOnOtherLane(Lane::NeighboringObjects &otherNeighbors) {
    return otherNeighbors.front == nullptr ||
           (otherNeighbors.front->x - x) >= (length / 2);
}

bool Car::hasBackSpaceOnOtherLane(Lane::NeighboringObjects &otherNeighbors) {
    return otherNeighbors.back == nullptr ||
           (x - otherNeighbors.back->x) >= (length / 2) + min_distance;
}