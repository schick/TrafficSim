//
// Created by oke on 07.12.18.
//

#include "model/Car.h"
#include "model/Road.h"
#include "NeighborFinder.h"
#include "IntelligentDriverModel.h"

void Car::nextStep(Lane::NeighboringObjects sameNeighbors) {

    auto lane = getLane();

    Road::NeighboringLanes neighboringLanes = lane->road->getNeighboringLanes(lane);

    auto leftNeighbors = NeighborFinder::getNeighboringObjects(neighboringLanes.left, this);
    double m_left = IntelligentDriverModel::getLaneChangeMetric(this, neighboringLanes.left, leftNeighbors, sameNeighbors);

    auto rightNeighbors = NeighborFinder::getNeighboringObjects(neighboringLanes.right, this);
    double m_right = IntelligentDriverModel::getLaneChangeMetric(this, neighboringLanes.right, rightNeighbors, sameNeighbors);

    if (m_left > 1 && m_left >= m_right) {
        // go to left lane
        new_acceleration = IntelligentDriverModel::getAcceleration(this, leftNeighbors.front);
        new_lane_offset = -1;
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        new_acceleration = IntelligentDriverModel::getAcceleration(this, rightNeighbors.front);
        new_lane_offset = 1;
    }
    else {
        // stay on lane
        new_acceleration = IntelligentDriverModel::getAcceleration(this, sameNeighbors.front);
        new_lane_offset = 0;
    }
}