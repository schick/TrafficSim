//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_CAR_H
#define PROJECT_CAR_H

#include <list>
#include <inttypes.h>

#include "TrafficObject.h"

class Car : public TrafficObject {

public:

     /**
     * data representing a turn at an intersection
     */
    enum TurnDirection {
        UTURN = 0,
        LEFT = 1,
        STRAIGHT = 2,
        RIGHT = 3
    };
    
    Car(int id, double length, double target_velocity, double max_acceleration, double target_deceleration,
            double min_distance, double target_headway, double politeness,
            double x=0, double v=0, double a=0)
                : target_velocity(target_velocity), max_acceleration(max_acceleration),
                    target_deceleration(target_deceleration), min_distance(min_distance),
                    target_headway(target_headway), politeness(politeness), lane(nullptr), TrafficObject(id, length, x, v, a) {}
    /**
     * properties
     */
    
    double target_velocity;
    double max_acceleration;
    double target_deceleration;
    double min_distance;
    //definition from ilias:
    const double min_s = 0.001;
    double target_headway;
    double politeness;
    double leftLaneAcceleration = 0;
    double sameLaneAcceleration = 0;
    double rightLaneAcceleration = 0;
    double new_acceleration = 0;
    int new_lane_offset  = 0;

    std::list<TurnDirection> turns;

    void nextStep();

    void calcSameLaneAcceleration(TrafficObject *leadingObject) override;
    double getSameLaneAcceleration() override;

    virtual void updateKinematicState();

    double getTravelledDistance();
    void setTravelledDistance(double value);

    /**
     * move this object to a specific lane.
     * @param lane lane to move object to
     */
    void moveToLane(Lane *lane);

    /**
     * remove object from any lane it may be assigned to
     */
    void removeFromLane();

    /**
     * get currently assigned lane
     * @return currently assigned lane
     */
    Lane *getLane();


private:

    double travelledDistance = 0.0;

    /**
     * current lane
     */
    Lane *lane;

};


#endif //PROJECT_CAR_H
