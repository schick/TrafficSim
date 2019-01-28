//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"

#include "IntelligentDriverModel.h"

void SequentialAlgorithm::prepareCars() {
    for (size_t i = 0; i < getRefScenario()->cars.size(); i++) {
        Car &car = getRefScenario()->cars[i];
        car.prepareNextMove();
    }
};

void SequentialAlgorithm::advanceCars() {
    for (Car &car : getRefScenario()->cars) {
        car.makeNextMove(*getRefScenario());
    }
}

void SequentialAlgorithm::advanceTrafficLights() {
    for (size_t i = 0; i < getRefScenario()->junctions.size(); i++) {
        Junction &junction = getRefScenario()->junctions[i];
        junction.updateSignals();
    }
}

void SequentialAlgorithm::sortLanes() {
    for (Lane &lane : getRefScenario()->lanes) {
        if (!lane.isSorted) {
            std::sort(lane.mTrafficObjects.begin(), lane.mTrafficObjects.end(), TrafficObject::Cmp());
            lane.isSorted = true;
        }
    }
}


void SequentialAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        sortLanes();
        prepareCars();
        advanceCars();
        advanceTrafficLights();
        getRefScenario()->current_step++;
    }
}
