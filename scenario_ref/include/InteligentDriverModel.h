
#ifndef PROJECT_INTELIGENT_DRIVER_MODEL
#define PROJECT_INTELIGENT_DRIVER_MODEL
#include <Car.h>

class InteligentDriverModel {
public:
    /**
     * calculate advance-data for next step
     * @return data representing the change
     */
    void nextStep(Car *car, Lane::NeighboringObjects neighbors);
    /**
     * advance car based of data
     * @param data data representing the change
     */
    void advanceStep(Car *car);

private:
    void updateKinematicState(Car *car);

    void updateLane(Car *car);

    void moveCarAcrossJunction(Car *car);

    bool isCarOverJunction(Car *car);

    double getLaneChangeMetricForLane(Car *car, Lane *neighboringLane, Lane::NeighboringObjects &neighbors, const Lane::NeighboringObjects &ownNeighbors);

    /**
     * calculate the desired acceleration. base calculation on leading object
     * @param leading_object leading object. may actually be in a different lane, this methods treats every object
     *      passed with this parameter as if it where in current lane
     * @return acceleration in m/s^2
     */
    double getAcceleration(Car *car, TrafficObject *leading_vehicle);

    /**
     * lane change metric described on slide 19 (22)
     * @param ownNeighbors neighbors on current lane
     * @param otherNeighbors neighbors on other lane
     * @return metric value in m/s^2
     */
    double laneChangeMetric(Car *car, const Lane::NeighboringObjects &ownNeighbors, Lane::NeighboringObjects &otherNeighbors);

    

    void setPosition(TrafficObject *trafficObject, double position);

    /**
     * get neighboring objects on this lane
     * @param object find neigbhoring objects for this object. may actually be on a different lane.
     *      this algorithm treats all objects as if there where on this lane.
     * @return neighboring objects
     */
    Lane::NeighboringObjects getNeighboringObjects(Lane *lane, TrafficObject *trafficObject);
}; 
#endif //PROJECT_INTELIGENT_DRIVER_MODEL