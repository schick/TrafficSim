//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm_id.h"

/*
std::vector<Car_id::AdvanceData> SequentialAlgorithm_id::calculateCarChanges() {

    std::vector<Car_id::AdvanceData> changes;
    for (auto &c : getIDScenario()->cars) {
        changes.emplace_back(c.nextStep(*getIDScenario()));
    }
    return changes;
};

void SequentialAlgorithm_id::advanceCars() {
    std::vector<Car_id::AdvanceData> changes = calculateCarChanges();
    for (Car_id::AdvanceData &d : changes) {
        getIDScenario()->cars[d.car].advanceStep(*getIDScenario(), d);
    }
}

void SequentialAlgorithm_id::advanceTrafficLights() {
    for (auto &j : getIDScenario()->junctions) {
        j.updateSignals(*getIDScenario());
    }
}


void SequentialAlgorithm_id::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        advanceCars();
        advanceTrafficLights();
    }
}
*/