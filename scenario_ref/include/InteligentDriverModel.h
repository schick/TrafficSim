#include <Car.h>

class InteligentDriverModel {
public:
    /**
     * calculate advance-data for next step
     * @return data representing the change
     */
    Car::AdvanceData nextStep(Car *car);
    /**
     * advance car based of data
     * @param data data representing the change
     */
    void advanceStep(Car::AdvanceData advancedCar, Car *car);

private:
    void updateKinematicState(Car::AdvanceData &data, Car *car);

    void updateLane(Car::AdvanceData &data, Car *car);

    void moveCarAcrossJunction(Car::AdvanceData &data, Car *car);

    bool isCarOverJunction(Car *car);

    double getLaneChangeMetricForLane(Car *car, Lane *neighboringLane, const Lane::NeighboringObjects &ownNeighbors);

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
    double laneChangeMetric(Car *car, Lane::NeighboringObjects ownNeighbors, Lane::NeighboringObjects otherNeighbors);

    

    void setPosition(TrafficObject *trafficObject, double position);
};