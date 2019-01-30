//
// Created by oke on 15.12.18.
//

#include "algorithms/OpenMPAlgorithm.h"

void OpenMPAlgorithm::prepareCars() {
    #pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->cars.size(); i++) {
        Car &car = getRefScenario()->cars[i];
        car.prepareNextMove();
    }
};

void OpenMPAlgorithm::advanceCars() {
    #pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->cars.size(); i++) {
        Car &car = getRefScenario()->cars[i];
        car.makeNextMove(*getRefScenario());
    }
}

void OpenMPAlgorithm::advanceTrafficLights() {
    #pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->junctions.size(); i++) {
        Junction &junction = getRefScenario()->junctions[i];
        junction.updateSignals();
    }
}

void OpenMPAlgorithm::sortLanes() {
    #pragma omp parallel for
    for (size_t i = 0; i < getRefScenario()->lanes.size(); i++) {
        Lane &lane = getRefScenario()->lanes[i];
        lane.sortCars();
    }
}

void OpenMPAlgorithm::cacheNeighbors() {
    #pragma omp parallel for
    for(int i = 0; i < getRefScenario()->roads.size(); i++) {
        getRefScenario()->roads[i].preCalcNeighbors();
    }
}

void OpenMPAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        sortLanes();
        cacheNeighbors();
        prepareCars();
        advanceCars();
        advanceTrafficLights();
        getRefScenario()->current_step++;
    }
}