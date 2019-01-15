//
// Created by oke on 07.12.18.
//

#include "model/Car.h"
#include "model/Road.h"
#include "NeighborFinder.h"
#include "IntelligentDriverModel.h"
#include "LaneChangeModel.h"

void Car::nextStep() {

    Lane *lane = getLane();

    Road::NeighboringLanes neighboringLanes = lane->road.getNeighboringLanes(lane);

    auto sameNeighbors = NeighborFinder::getNeighboringObjects(lane, this);
    calcSameLaneAcceleration(sameNeighbors.front);

    auto leftNeighbors = NeighborFinder::getNeighboringObjects(neighboringLanes.left, this);
    double m_left = LaneChangeModel::getLaneChangeMetric(this, sameNeighbors, neighboringLanes.left, leftNeighbors, true);

    auto rightNeighbors = NeighborFinder::getNeighboringObjects(neighboringLanes.right, this);
    double m_right = LaneChangeModel::getLaneChangeMetric(this, sameNeighbors, neighboringLanes.right, rightNeighbors, false);

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