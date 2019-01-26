//
// Created by oke on 07.12.18.
//

#include "model/Car.h"
#include "model/Road.h"
#include "NeighborFinder.h"
#include "IntelligentDriverModel.h"
#include "LaneChangeModel.h"

#include <algorithm>

void Car::nextStep() {

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
        new_acceleration = leftLaneAcceleration;
        new_lane_offset = -1;
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        new_acceleration = rightLaneAcceleration;
        new_lane_offset = 1;
    }
    else {
        // stay on lane
        new_acceleration = sameLaneAcceleration;
        new_lane_offset = 0;
    }
}

void Car::calcSameLaneAcceleration(TrafficObject *leadingObject) {
    sameLaneAcceleration = IntelligentDriverModel::getAcceleration(this, leadingObject);
}

double Car::getSameLaneAcceleration() {
    return sameLaneAcceleration;
}

void Car::updateKinematicState() {
    a = new_acceleration;
    v = std::max(v + a, 0.);
    setPosition(getPosition() + v);
    travelledDistance += v;
}

double Car::getTravelledDistance() {
    return travelledDistance;
}

void Car::setTravelledDistance(double value) {
    travelledDistance = value;
}

void Car::moveToLane(Lane *lane) {
    if (lane != nullptr) {
        removeFromLane();
    }
    this->lane = lane;
    std::lock_guard<std::mutex> lock(lane->laneLock);
    lane->mTrafficObjects.push_back(this);
    lane->isSorted = false;
}

void Car::removeFromLane() {

    if (lane == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(lane->laneLock);
    auto position = std::find(lane->mTrafficObjects.rbegin(), lane->mTrafficObjects.rend(), this);
    std::iter_swap(position, lane->mTrafficObjects.end() - 1);
    lane->mTrafficObjects.pop_back();
    lane->isSorted = false;
    lane = nullptr;
}

Lane *Car::getLane() {
    return lane;
}