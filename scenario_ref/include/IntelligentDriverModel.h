
#ifndef PROJECT_INTELIGENT_DRIVER_MODEL
#define PROJECT_INTELIGENT_DRIVER_MODEL
#include <model/Car.h>

class IntelligentDriverModel {
public:
    /**
     * advance car based of data
     * @param data data representing the change
     */
    static void advanceStep(Car *car);

    /**
     * calculate the desired acceleration. base calculation on leading object
     * @param leading_object leading object. may actually be in a different lane, this methods treats every object
     *      passed with this parameter as if it where in current lane
     * @return acceleration in m/s^2
     */
    static double getAcceleration(Car *car, TrafficObject *leading_vehicle);

    /**
     * lane change metric described on slide 19 (22)
     * @param car the relevant car
     * @param neighboringLane the other lane
     * @param ownNeighbors neighbors on current lane
     * @param otherNeighbors neighbors on other lane
     * @return metric value in m/s^2
     */
    static double getLaneChangeMetric(Car *car, Lane *neighboringLane, Lane::NeighboringObjects &ownNeighbors, Lane::NeighboringObjects &otherNeighbors);

private:
    static void updateKinematicState(Car *car);

    static void updateLane(Car *car);

    static void moveCarAcrossJunction(Car *car);

    static bool isCarOverJunction(Car *car);

    static double laneChangeMetric(Car *car, const Lane::NeighboringObjects &ownNeighbors, Lane::NeighboringObjects &otherNeighbors);

};


#endif //PROJECT_INTELIGENT_DRIVER_MODEL