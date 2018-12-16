//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"


std::vector<Car::AdvanceData> SequentialAlgorithm::calculateCarChanges() {

    std::vector<Car::AdvanceData> changes;
    for (std::unique_ptr<Car> &c : getScenario()->cars) {
        changes.emplace_back(c->nextStep());
    }
    return changes;
};

void SequentialAlgorithm::advanceCars() {
    std::vector<Car::AdvanceData> changes = calculateCarChanges();
    for (Car::AdvanceData &d : changes) {
        d.car->advanceStep(d);
    }
}

void SequentialAlgorithm::advanceTrafficLights() {
    for (std::unique_ptr<Junction> &j : getScenario()->junctions) {
        j->updateSignals();
    }
}


void SequentialAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        advanceCars();
        advanceTrafficLights();
    }
}
