//
// Created by maxi on 1/8/19.
//

#include "LaneChangeModel.h"

double LaneChangeModel::getLaneChangeMetric(Car *car, Lane::NeighboringObjects &sameNeighbors, Lane *otherLane,
                                            Lane::NeighboringObjects &otherNeighbors, bool isLeftLane) {
    if (otherLane == nullptr) {
        return 0;
    } else {
        return calculateLaneChangeMetric(car, sameNeighbors, otherNeighbors, isLeftLane);
    }
}

double LaneChangeModel::calculateLaneChangeMetric(Car *car, Lane::NeighboringObjects &sameNeighbors,
                                                  Lane::NeighboringObjects &otherNeighbors, bool isLeftLane) {

    if (hasFrontSpaceOnOtherLane(car, otherNeighbors) && hasBackSpaceOnOtherLane(car, otherNeighbors)) {

        double sameLaneAcceleration = car->getSameLaneAcceleration();
        double otherLaneAcceleration = IntelligentDriverModel::getAcceleration(car, otherNeighbors.front);

        if (otherLaneAcceleration > sameLaneAcceleration) {

            if (isLeftLane) {
                car->leftLaneAcceleration = otherLaneAcceleration;
            } else {
                car->rightLaneAcceleration = otherLaneAcceleration;
            }

            double other_lane_diff = 0;
            if (otherNeighbors.back != nullptr) {
                other_lane_diff = (
                        IntelligentDriverModel::getAcceleration(dynamic_cast<Car *>(otherNeighbors.back), car) -
                        IntelligentDriverModel::getAcceleration(dynamic_cast<Car *>(otherNeighbors.back),
                                                                otherNeighbors.front));
            }

            double behind_diff = 0;
            if (sameNeighbors.back != nullptr) {
                behind_diff = (IntelligentDriverModel::getAcceleration(dynamic_cast<Car *>(sameNeighbors.back),
                                                                       sameNeighbors.front) -
                               IntelligentDriverModel::getAcceleration(dynamic_cast<Car *>(sameNeighbors.back), car));
            }

            return otherLaneAcceleration - sameLaneAcceleration + car->politeness * (behind_diff + other_lane_diff);
        }
    }
    return 0;
}

bool LaneChangeModel::hasFrontSpaceOnOtherLane(Car *car, Lane::NeighboringObjects &otherNeighbors) {
    return otherNeighbors.front == nullptr ||
           (otherNeighbors.front->getPosition() - car->getPosition()) >= (car->length / 2);
}

bool LaneChangeModel::hasBackSpaceOnOtherLane(Car *car, Lane::NeighboringObjects &otherNeighbors) {
    return otherNeighbors.back == nullptr ||
           (car->getPosition() - otherNeighbors.back->getPosition()) >= (car->length / 2) + car->min_distance;
}