
#ifndef PROJECT_INTELIGENT_DRIVER_MODEL
#define PROJECT_INTELIGENT_DRIVER_MODEL


#include <model/Car.h>

class IntelligentDriverModel {
public:
    /**
     * advance car based of data
     * @param data data representing the change
     */
    static void advanceStep(Car &car);

    /**
     * calculate the desired acceleration. base calculation on leading object
     * @param leading_object leading object. may actually be in a different lane, this methods treats every object
     *      passed with this parameter as if it where in current lane
     * @return acceleration in m/s^2
     */
    static double getAcceleration(Car *car, TrafficObject *leading_vehicle);


private:
    static void updateLane(Car &car);

    static void moveCarAcrossJunction(Car &Car);

    static bool isCarOverJunction(Car &car);
};


#endif //PROJECT_INTELIGENT_DRIVER_MODEL