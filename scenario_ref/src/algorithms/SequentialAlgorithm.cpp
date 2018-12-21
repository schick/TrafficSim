//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"


std::vector<Car::AdvanceData> SequentialAlgorithm::calculateCarChanges() {
    
    
    std::vector<Car::AdvanceData> changes;
    for (std::unique_ptr<Car> &c : getRefScenario()->cars) {
        changes.emplace_back(idm.nextStep(c.get()));
    }
    return changes;
};

void SequentialAlgorithm::advanceCars() {
    std::vector<Car::AdvanceData> changes = calculateCarChanges();
    for (Car::AdvanceData &d : changes) {
        idm.advanceStep(d, d.car);
    }
}

void SequentialAlgorithm::advanceTrafficLights() {
    for (std::unique_ptr<Junction> &j : getRefScenario()->junctions) {
        j->updateSignals();
    }
}

void SequentialAlgorithm::sortLanes() {
    for (auto &lane : getRefScenario()->lanes) {
        //auto trafficObjects = lane->mTrafficObjects;
        std::sort(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), TrafficObject::Cmp());
    }
}


void SequentialAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        sortLanes();
        advanceCars();
        advanceTrafficLights();
    }
}
