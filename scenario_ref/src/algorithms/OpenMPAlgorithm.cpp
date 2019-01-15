//
// Created by oke on 15.12.18.
//

#include "algorithms/OpenMPAlgorithm.h"


void OpenMPAlgorithm::calculateCarChanges() {
#pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->cars.size(); i++) {
        Car &car = getRefScenario()->cars[i];
        car.nextStep();
    }
};

void OpenMPAlgorithm::advanceCars() {
#pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->cars.size(); i++) {
        Car &car = getRefScenario()->cars[i];
        IntelligentDriverModel::advanceStep(car);
    }
}

void OpenMPAlgorithm::advanceTrafficLights() {
#pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->junctions.size(); i++) {
        Junction &junction = getRefScenario()->junctions.at(i);
        junction.updateSignals();
    }
}

void OpenMPAlgorithm::sortLanesAndCalculateAcceleration() {
#pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->lanes.size(); i++) {
        Lane &lane = getRefScenario()->lanes.at(i);
        if (!lane.isSorted) {
            std::sort(lane.mTrafficObjects.begin(), lane.mTrafficObjects.end(), TrafficObject::Cmp());
            lane.isSorted = true;
        }
    }
}

void OpenMPAlgorithm::advance(size_t steps) {

    for (int i = 0; i < steps; i++) {
        sortLanesAndCalculateAcceleration();
        calculateCarChanges();
        advanceCars();
        advanceTrafficLights();
    }
}