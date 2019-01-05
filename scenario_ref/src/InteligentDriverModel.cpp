#include "InteligentDriverModel.h"
#include <model/Car.h>
#include <model/Road.h>
#include <assert.h>
#include <math.h>

void InteligentDriverModel::nextStep(Car *car, Lane::NeighboringObjects neighbors) {
    if (car == nullptr) {
        return;
    }
    auto lane = car->getLane();
    Lane::NeighboringObjects ownNeighbors = getNeighboringObjects(lane, car);
    Road::NeighboringLanes neighboringLanes = lane->road->getNeighboringLanes(lane);

    auto leftNeighbors = getNeighboringObjects(neighboringLanes.left, car);
    double m_left = getLaneChangeMetricForLane(car, neighboringLanes.left, leftNeighbors, ownNeighbors);

    auto rightNeighbors = getNeighboringObjects(neighboringLanes.right, car);
    double m_right = getLaneChangeMetricForLane(car, neighboringLanes.right, rightNeighbors, ownNeighbors);

    if (m_left > 1 && m_left >= m_right) {
        // go to left lane
        car->new_acceleration = getAcceleration(car, leftNeighbors.front);
        car->new_lane_offset = -1;
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        car->new_acceleration = getAcceleration(car, rightNeighbors.front);
        car->new_lane_offset = 1;
    }
    else {
        // stay on lane
        car->new_acceleration = getAcceleration(car, ownNeighbors.front);
        car->new_lane_offset = 0;
    }
}

void InteligentDriverModel::advanceStep(Car *car) {
    if (car == nullptr) {
        return;
    }
    updateKinematicState(car);
    updateLane(car);
}

void InteligentDriverModel::updateKinematicState(Car *car) {
    //assert(data.car == car);
    car->a = car->new_acceleration;
    car->v = std::max(car->v + car->a, 0.);
    setPosition(car, car->getPosition() + car->v);
}

void InteligentDriverModel::updateLane(Car *car) {
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

void InteligentDriverModel::moveCarAcrossJunction(Car *car) {
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

bool InteligentDriverModel::isCarOverJunction(Car *car) {
    return car->getPosition() >= car->getLane()->getLength();
}

double InteligentDriverModel::getLaneChangeMetricForLane(Car *car, Lane *neighboringLane, Lane::NeighboringObjects &neighbors, const Lane::NeighboringObjects &ownNeighbors) {
    if (neighboringLane != nullptr) {
        return laneChangeMetric(car, ownNeighbors, neighbors);
    }
    return 0;
}

double InteligentDriverModel::getAcceleration(Car *car, TrafficObject *leading_vehicle) {
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

double InteligentDriverModel::laneChangeMetric(Car *car, const Lane::NeighboringObjects &ownNeighbors, Lane::NeighboringObjects &otherNeighbors) {

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

void InteligentDriverModel::setPosition(TrafficObject *car, double position) {
    car->setPosition(position);    
}

Lane::NeighboringObjects InteligentDriverModel::getNeighboringObjects(Lane *lane, TrafficObject *trafficObject) {
    //create empty neighboringObjects struct
    auto neighboringObjects = Lane::NeighboringObjects();

    //if null reference return empty struct
    if (lane == nullptr)
        return neighboringObjects;

    auto trafficObjects = lane->mTrafficObjects;
    //if lane is empty return empty struct
    if (trafficObjects.size() == 0)
        return Lane::NeighboringObjects();

    auto it = std::lower_bound(trafficObjects.begin(), trafficObjects.end(), trafficObject, TrafficObject::Cmp());
    Lane::NeighboringObjects result;

    if (it != trafficObjects.begin())
        result.back = *(it - 1);

    if (trafficObjects.end() == it || *it != trafficObject) {
        if (it != trafficObjects.end())
            result.front = *it;
    }
    else {
        if (it + 1 != trafficObjects.end())
            result.front = *(it + 1);
    }
    return result;
}

