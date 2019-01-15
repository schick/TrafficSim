//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"


void SequentialAlgorithm::calculateCarChanges() {
#pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->cars.size(); i++) {
        Car &car = getRefScenario()->cars[i];
        car.nextStep();
    }
};

void SequentialAlgorithm::advanceCars() {
    for (Car &car : getRefScenario()->cars) {
        IntelligentDriverModel::advanceStep(car);
    }
}

void SequentialAlgorithm::advanceTrafficLights() {
    for (auto pair : getRefScenario()->junctions) {
        pair.second->updateSignals();
    }
}

void SequentialAlgorithm::sortLanes() {
    for (auto &lane : getRefScenario()->lanes) {
        if (!lane.isSorted) {
            std::sort(lane.mTrafficObjects.begin(), lane.mTrafficObjects.end(), TrafficObject::Cmp());
            lane.isSorted = true;
        }
    }
}


void SequentialAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        sortLanes();
        calculateCarChanges();
        advanceCars();
        advanceTrafficLights();
    }
}
