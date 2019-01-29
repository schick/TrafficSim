//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_CAR_H
#define PROJECT_CAR_H

#include <list>

#include "TrafficObject.h"
#include "TrafficLight.h"
#include "Lane.h"

class Scenario;

class Car : public TrafficObject {

public:

    /**
     * compare object to compare Traffic objects by
     */
    struct Cmp {
        bool operator () (const Car *lhs, Car *rhs) {
            if (lhs->x == rhs->x)
                return lhs->id > rhs->id;
            return lhs->x < rhs->x;
        }
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
    
    Car(uint64_t id, double length, double target_velocity, double max_acceleration, double target_deceleration,
            double min_distance, double target_headway, double politeness, double x=0, double v=0, double a=0);

    // definition from ilias:
    const double min_s = 0.001;

    /**
     * properties
     */
    double target_velocity;
    double max_acceleration;
    double target_deceleration;
    double min_distance;
    double target_headway;
    double politeness;
    std::list<TurnDirection> turns;

    double travelledDistance = 0.0;

    /**
     * temporary values (advance data)
     */
     struct AdvanceData {
         double leftLaneAcceleration = 0;
         double sameLaneAcceleration = 0;
         double rightLaneAcceleration = 0;
         double new_acceleration = 0;
         int new_lane_offset = 0;
     } advance_data;


    void prepareNextMove();

    void makeNextMove(Scenario &scenario);

    void updateKinematicState();

    /**
     * move this object to a specific lane.
     * @param lane lane to move object to
     */
    void moveToLane(Lane &lane);

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

    void calcSameLaneAcceleration(TrafficObject *leadingObject);

    /**
     * current lane
     */
    Lane *lane;

    void updateLane(Scenario &scenario);
    void moveCarAcrossJunction(Scenario &scenario);
    bool isCarOverJunction();
    double getAcceleration(TrafficObject *leading_vehicle);
    double calculateWithLead(TrafficObject &leading_vehicle);
    void setLeadingTrafficObject(TrafficObject * &leading_vehicle, TrafficLight &trafficLight);


    double getLaneChangeMetric(Lane::NeighboringObjects &sameNeighbors, Lane *otherLane,
            Lane::NeighboringObjects &otherNeighbors, bool isLeftLane);
    double calculateLaneChangeMetric(Lane::NeighboringObjects &sameNeighbors,
            Lane::NeighboringObjects &otherNeighbors, bool isLeftLane);
    bool hasFrontSpaceOnOtherLane(Lane::NeighboringObjects &otherNeighbors);
    bool hasBackSpaceOnOtherLane(Lane::NeighboringObjects &otherNeighbors);
};


#endif //PROJECT_CAR_H
