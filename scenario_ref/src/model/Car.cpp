//
// Created by oke on 07.12.18.
//

#include "model/Car.h"
#include "model/Road.h"
#include "NeighborFinder.h"
#include "IntelligentDriverModel.h"
#include "LaneChangeModel.h"

#include <algorithm>


Car::Car(uint64_t id, double length, double target_velocity, double max_acceleration, double target_deceleration,
    double min_distance, double target_headway, double politeness, double x, double v, double a)
        : target_velocity(target_velocity), max_acceleration(max_acceleration), target_deceleration(target_deceleration),
            min_distance(min_distance), target_headway(target_headway), politeness(politeness), lane(nullptr),
            TrafficObject(id, length, x, v, a) {}

void Car::prepareNextMove() {

    Lane *lane = getLane();

    Road::NeighboringLanes neighboringLanes = lane->road.getNeighboringLanes(lane);

    auto sameNeighbors = NeighborFinder::getNeighboringObjects(lane, this);
    calcSameLaneAcceleration(sameNeighbors.front);

    auto leftNeighbors = NeighborFinder::getNeighboringObjects(neighboringLanes.left, this);
    double m_left = LaneChangeModel::getLaneChangeMetric(*this, sameNeighbors, neighboringLanes.left, leftNeighbors, true);

    auto rightNeighbors = NeighborFinder::getNeighboringObjects(neighboringLanes.right, this);
    double m_right = LaneChangeModel::getLaneChangeMetric(*this, sameNeighbors, neighboringLanes.right, rightNeighbors, false);

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
    IntelligentDriverModel::updateLane(*this, scenario);
}

void Car::calcSameLaneAcceleration(TrafficObject *leadingObject) {
    advance_data.sameLaneAcceleration = IntelligentDriverModel::getAcceleration(this, leadingObject);
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