//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_CAR_H
#define PROJECT_CAR_H

#include <vector>
#include <inttypes.h>

#include "TrafficObject.h"
#include "Lane.h"

class Car : public TrafficObject {

private:

    /**
     * lane change metric described on slide 19 (22)
     * @param ownNeighbors neighbors on current lane
     * @param otherNeighbors neighbors on other lane
     * @return metric value in m/s^2
     */
    double laneChangeMetic(Lane::NeighboringObjects ownNeighbors, Lane::NeighboringObjects otherNeighbors);

public:

    /**
     * Data representing next advance of a car
     */
    struct AdvanceData {
        AdvanceData(Car *car, double acceleration, Lane* lane_change)
            : car(car), acceleration(acceleration), lane_change(lane_change) {};
        Car *car = nullptr;
        double acceleration = 0;
        Lane* lane_change = nullptr;
    };


    /**
     * data representing a turn at an intersection
     */
    enum TurnDirection {
        UTURN = 0,
        RIGHT = 1,
        STRAIGHT = 2,
        LEFT = 3
    };

    /**
     * some constructor
     * @param id
     * @param length
     * @param target_velocity
     * @param max_acceleration
     * @param target_deceleration
     * @param min_distance
     * @param target_headway
     * @param politeness
     * @param x
     * @param v
     * @param a
     */
    Car(uint64_t id, double length, double target_velocity, double max_acceleration, double target_deceleration,
            double min_distance, double target_headway, double politeness,
            double x=0, double v=0, double a=0)
                : id(id), length(length), target_velocity(target_velocity), max_acceleration(max_acceleration),
                    target_deceleration(target_deceleration), min_distance(min_distance),
                    target_headway(target_headway), politeness(politeness), TrafficObject(length, x, v, a) {}
    /**
     * properties
     */
    uint64_t id;
    double length;
    double target_velocity;
    double max_acceleration;
    double target_deceleration;
    double min_distance;
    double target_headway;
    double politeness;

    std::vector<TurnDirection> turns;

    /**
     * calculate advance-data for next step
     * @return data representing the change
     */
    AdvanceData nextStep();

    /**
     * advance car based of data
     * @param data data representing the change
     */
    void advanceStep(AdvanceData data);

    /**
     * calculate the desired acceleration. base calculation on leading object
     * @param leading_object leading object. may actually be in a different lane, this methods treats every object
     *      passed with this parameter as if it where in current lane
     * @return acceleration in m/s^2
     */
    double getAcceleration(TrafficObject *leading_object) override;


};


#endif //PROJECT_CAR_H
