//
// Created by oke on 15.12.18.
//

#include "by_id/algorithms/SequentialAlgorithm.h"


std::vector<Car::AdvanceData> SequentialAlgorithm::calculateCarChanges() {

    std::vector<Car::AdvanceData> changes;
    for (auto &c : getScenario()->cars) {
        changes.emplace_back(c.nextStep(*getScenario()));
    }
    return changes;
};

void SequentialAlgorithm::advanceCars() {
    std::vector<Car::AdvanceData> changes = calculateCarChanges();
    for (Car::AdvanceData &d : changes) {
        getScenario()->cars[d.car].advanceStep(*getScenario(), d);
    }
}

void SequentialAlgorithm::advanceTrafficLights() {
    for (auto &j : getScenario()->junctions) {
        j.updateSignals(*getScenario());
    }
}


void SequentialAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        advanceCars();
        advanceTrafficLights();
    }
}
