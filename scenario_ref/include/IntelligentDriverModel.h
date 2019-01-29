
#ifndef PROJECT_INTELIGENT_DRIVER_MODEL
#define PROJECT_INTELIGENT_DRIVER_MODEL


#include <model/Car.h>
#include "model/Scenario.h"

class IntelligentDriverModel {
public:
    /**
     * calculate the desired acceleration. base calculation on leading object
     * @param leading_object leading object. may actually be in a different lane, this methods treats every object
     *      passed with this parameter as if it where in current lane
     * @return acceleration in m/s^2
     */
    static double getAcceleration(Car *car, TrafficObject *leading_vehicle);

    static void updateLane(Car &car, Scenario &scenario);

private:
    static void moveCarAcrossJunction(Car &Car, Scenario &scenario);

    static bool isCarOverJunction(Car &car);

    static void setLeadingTrafficObject(TrafficObject *&leading_vehicle, Car &car, TrafficLight &trafficLight);
    
    static double calculateWithLead(Car &car, TrafficObject &leading_vehicle);
};


#endif //PROJECT_INTELIGENT_DRIVER_MODEL