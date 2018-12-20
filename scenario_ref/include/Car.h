//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_CAR_H
#define PROJECT_CAR_H

#include <list>
#include <inttypes.h>

#include "TrafficObject.h"
#include "Lane.h"

class Car : public TrafficObject {

public:

    /**
     * Data representing next advance of a car
     */
    struct AdvanceData {
        AdvanceData() = default;
        AdvanceData(Car *car, double acceleration, int lane_offset)
            : car(car), acceleration(acceleration), lane_offset(lane_offset) {};
        Car *car = nullptr;
        double acceleration = 0;
        int8_t lane_offset = 0;
    };


    /**
     * data representing a turn at an intersection
     */
    enum TurnDirection {
        UTURN = 0,
        LEFT = 1,
        STRAIGHT = 2,
        RIGHT = 3
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
    Car(int id, double length, double target_velocity, double max_acceleration, double target_deceleration,
            double min_distance, double target_headway, double politeness,
            double x=0, double v=0, double a=0)
                : length(length), target_velocity(target_velocity), max_acceleration(max_acceleration),
                    target_deceleration(target_deceleration), min_distance(min_distance),
                    target_headway(target_headway), politeness(politeness), TrafficObject(id, length, x, v, a) {}
    /**
     * properties
     */
    
    double length;
    double target_velocity;
    double max_acceleration;
    double target_deceleration;
    double min_distance;
    //definition from ilias:
    const double min_s = 0.001;
    double target_headway;
    double politeness;


    std::list<TurnDirection> turns;

};


#endif //PROJECT_CAR_H
