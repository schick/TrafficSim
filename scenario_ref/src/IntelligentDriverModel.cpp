#include "IntelligentDriverModel.h"
#include <model/Car.h>
#include <model/Road.h>
#include <assert.h>
#include <math.h>

void IntelligentDriverModel::advanceStep(Car *car) {
    if (car == nullptr) {
        return;
    }
    updateKinematicState(car);
    updateLane(car);
}

void IntelligentDriverModel::updateKinematicState(Car *car) {
    //assert(data.car == car);
    car->a = car->new_acceleration;
    car->v = std::max(car->v + car->a, 0.);
    setPosition(car, car->getPosition() + car->v);
}

void IntelligentDriverModel::updateLane(Car *car) {
    //assert(data.car == car);

    // check for junction
    if (isCarOverJunction(car)) {
        moveCarAcrossJunction(car);
    }
    else {
        // just do a lane change if wanted
        if (car->new_lane_offset != 0) {
            // lane_offset should be validated in this case
            assert(car->getLane()->road->lanes.size() > car->getLane()->lane + car->new_lane_offset);
            car->moveToLane(car->getLane()->road->lanes[car->getLane()->lane + car->new_lane_offset]);
        }
    }
}

void IntelligentDriverModel::moveCarAcrossJunction(Car *car) {
    assert(!car->turns.empty());

    Lane *old_lane = car->getLane();

    // subtract moved position on current lane from distance
    auto oldLaneLength = old_lane->road->getLength();
    setPosition(car, car->getPosition() - oldLaneLength);

    // select direction based on current direction and turn
    int direction = (old_lane->road->getDirection() + car->turns.front() + 2) % 4;

    // if no road in that direction -> select next to the right
    Road *nextRoad;
    while ((nextRoad = old_lane->road->to->outgoing[direction]) == nullptr) direction = (++direction) % 4;

    // move car to same or the right lane AFTER lane change
    int indexOfNextLane = std::min((int) nextRoad->lanes.size() - 1, old_lane->lane + car->new_lane_offset);
    indexOfNextLane = std::max(0, indexOfNextLane);

    car->moveToLane(nextRoad->lanes[indexOfNextLane]);

    // update next turns
    car->turns.push_back(car->turns.front());
    car->turns.pop_front();
}

bool IntelligentDriverModel::isCarOverJunction(Car *car) {
    return car->getPosition() >= car->getLane()->getLength();
}

double IntelligentDriverModel::getLaneChangeMetric(Car *car, Lane *neighboringLane, Lane::NeighboringObjects &neighbors, Lane::NeighboringObjects &ownNeighbors) {
    if (neighboringLane != nullptr) {
        return laneChangeMetric(car, ownNeighbors, neighbors);
    }
    return 0;
}

double IntelligentDriverModel::getAcceleration(Car *car, TrafficObject *leading_vehicle) {
    if (car == nullptr) {
        return 0;
    }
    double vel_fraction = (car->v / std::min(car->getLane()->road->limit, car->target_velocity));
    double without_lead = 1. - vel_fraction * vel_fraction * vel_fraction * vel_fraction; // faster than pow

    double with_lead = 0;
    if (leading_vehicle != nullptr) {
        double delta_v = car->v - leading_vehicle->v;
        double s = std::max(leading_vehicle->getPosition() - car->getPosition() - leading_vehicle->length, car->min_s);
        with_lead = (car->min_distance + car->v * car->target_headway +
            (car->v * delta_v) / (2. * sqrt(car->max_acceleration * car->target_deceleration))) / s;
        with_lead = with_lead * with_lead; // faster than pow
    }
    double acceleration = car->max_acceleration * (without_lead - with_lead);
    return acceleration;
}

double IntelligentDriverModel::laneChangeMetric(Car *car, const Lane::NeighboringObjects &ownNeighbors, Lane::NeighboringObjects &otherNeighbors) {

    if ((otherNeighbors.front == nullptr || (otherNeighbors.front->getPosition() - car->getPosition()) >= (car->length / 2)) &&
        (otherNeighbors.back == nullptr || (car->getPosition() - otherNeighbors.back->getPosition()) >= (car->length / 2) + car->min_distance)) {
        double own_wo_lc = getAcceleration(car, ownNeighbors.front);
        double own_w_lc = getAcceleration(car, otherNeighbors.front);

        double other_lane_diff = 0;
        if (otherNeighbors.back != nullptr) {
            other_lane_diff = (getAcceleration(dynamic_cast<Car *>(otherNeighbors.back), car) -
               getAcceleration(dynamic_cast<Car *>(otherNeighbors.back), otherNeighbors.front));
        }


        double behind_diff = 0;
        if (ownNeighbors.back != nullptr) {
            behind_diff = (getAcceleration(dynamic_cast<Car *>(ownNeighbors.back), ownNeighbors.front) -
                getAcceleration(dynamic_cast<Car *>(ownNeighbors.back), car));
        }

        if (own_w_lc > own_wo_lc) {
            return own_w_lc - own_wo_lc + car->politeness * (behind_diff + other_lane_diff);
        }
    }
    return 0;
}

void IntelligentDriverModel::setPosition(TrafficObject *car, double position) {
    car->setPosition(position);    
}