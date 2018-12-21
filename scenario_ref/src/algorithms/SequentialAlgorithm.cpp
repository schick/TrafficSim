//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"


std::vector<Car::AdvanceData> SequentialAlgorithm::calculateCarChanges() {

    std::vector<Car::AdvanceData> changes;
    for (std::shared_ptr<Car> &c : getRefScenario()->cars) {
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
    for (std::shared_ptr<Junction> &j : getRefScenario()->junctions) {
        j->updateSignals();
    }
}


void SequentialAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        advanceCars();
        advanceTrafficLights();
    }
}
