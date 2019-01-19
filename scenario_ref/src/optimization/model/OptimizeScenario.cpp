//
// Created by maxi on 1/18/19.
//

#include "optimization/model/OptimizeScenario.h"

void OptimizeScenario::parse(nlohmann::json &input) {
    parseJunctions(input);
    parseRoads(input);
    parseCars(input);
}

json OptimizeScenario::toJson() {
    json output;
    for (Junction &junction : junctions) {
        json out_junction;

        out_junction["id"] = junction.id;

        for (Junction::Signal &signal : junction.signals) {
            json out_signal;

            out_signal["dir"] = static_cast<int>(signal.direction);
            out_signal["time"] = signal.duration;

            out_junction["signals"].push_back(out_signal);
        }

        output["junctions"].push_back(out_junction);
    }
    return output;
}

double OptimizeScenario::getTraveledDistance() {
    double sum;
    for (Car &car : cars) {
        sum += car.getTraveledDistance();
    }
    return sum;
}

void OptimizeScenario::parseSignals(const json &input) {
    //Do nothing
}
