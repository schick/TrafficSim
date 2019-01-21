//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_CAR_ID_H
#define PROJECT_CAR_ID_H

#include "TrafficObject_id.h"
class CudaScenario_id;
class Car_id : public TrafficObject_id {

private:

    /**
     * lane change metric described on slide 19 (22)
     * @param ownNeighbors neighbors on current lane
     * @param otherNeighbors neighbors on other lane
     * @return metric value in m/s^2
     */
    double laneChangeMetric(Scenario_id &s, size_t other_front, size_t other_back, size_t own_front, size_t own_back);

public:

    /**
     * properties
     */
    double length;
    double target_velocity;
    double max_acceleration;
    double target_deceleration;
    double min_distance;
    double target_headway;
    double politeness;

    size_t turns_begin;
    size_t turns_count;
    size_t current_turn_offset;
    double travelled_distance;

    //definition from ilias:
    static constexpr double min_s = 0.001;

    /**
     * Data representing next advance of a car
     */
    struct AdvanceData {
        AdvanceData() = default;
        CUDA_HOSTDEV AdvanceData(size_t car, double acceleration, int8_t lane_offset)
            : car(car), acceleration(acceleration), lane_offset(lane_offset) {};
        size_t car = (size_t )-1;
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
    explicit Car_id(size_t id=(size_t)-1, double length=0, double target_velocity=0, double max_acceleration=0,
            double target_deceleration=0, double min_distance=0, double target_headway=0, double politeness=0,
                    size_t turns_begin=(size_t )-1, size_t turns_count=0, size_t lane=(size_t)-1, double x=0, double v=0, double a=0)
                : length(length), target_velocity(target_velocity), max_acceleration(max_acceleration),
                    target_deceleration(target_deceleration), min_distance(min_distance),
                    target_headway(target_headway), politeness(politeness), turns_begin(turns_begin),
                    turns_count(turns_count), current_turn_offset(0), TrafficObject_id(id, length, lane, x, v, a),
                    travelled_distance(0) {}

    /**
     * calculate advance-data for next step
     * @return data representing the change
     */
    AdvanceData nextStep(Scenario_id &s, size_t own_front, size_t own_back,
            size_t left_front, size_t left_back,
            size_t right_front, size_t right_back);

    
    double getLaneChangeMetricForLane(Scenario_id &s, int neighboringLane, const Lane_id::NeighboringObjects &ownNeighbors);

    /**
     * advance car based of data
     * @param data data representing the change
     */
    void advanceStep(Scenario_id &s, AdvanceData &data);

    /**
     * calculate the desired acceleration. base calculation on leading object
     * @param leading_object leading object. may actually be in a different lane, this methods treats every object
     *      passed with this parameter as if it where in current lane
     * @return acceleration in m/s^2
     */
    double getAcceleration(Scenario_id &s, size_t leading_object_id) override;

private:

    void updateLane(Scenario_id &s, AdvanceData &data);

    bool isCarOverJunction(Scenario_id &s);

    void moveCarAcrossJunction(Scenario_id &s, Car_id::AdvanceData &data);

    void updateKinematicState(Car_id::AdvanceData &data);
};


#endif //PROJECT_CAR_H
