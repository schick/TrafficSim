//
// Created by maxi on 1/18/19.
//

#include "optimization/model/OptimizeScenario.h"

double OptimizeScenario::getTraveledDistance() {
    double sum;
    for (Car &car : cars) {
        sum += car.getTraveledDistance();
    }
    return sum;
}

void OptimizeScenario::parse(nlohmann::json &input) {
    parseJunctions(input);
    parseRoads(input);
    parseCars(input);
}

void OptimizeScenario::parseSignals(const json &input) {
    //Do nothing
}
