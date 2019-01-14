//
// Created by oke on 15.12.18.
//

#include "algorithms/OpenMPAlgorithm.h"


void OpenMPAlgorithm::calculateCarChanges() {
#pragma omp for
    for (long j = 0; j < getRefScenario()->roads.size(); j++) {
        auto r = getRefScenario()->roads.at(j);
        for (auto &l : r.lanes) {
            for (long i = 0; i < l->mTrafficObjects.size(); i++) {
                //Iterate over cars of lane. neighbors are it+1 and it-1.
                Lane::NeighboringObjects neighbors;

                //set preceding car for all cars except the first one
                if (i != 0)
                    neighbors.back = l->mTrafficObjects.at(i - 1);

                //set next car for all cars except the last one
                if (i != l->mTrafficObjects.size() - 1)
                    neighbors.front = l->mTrafficObjects.at(i + 1);

                l->mTrafficObjects.at(i)->nextStep(neighbors);
            }
        }
    }
};

void OpenMPAlgorithm::advanceCars() {
#pragma omp for
    for (long i = 0; i < getRefScenario()->cars.size(); i++) {
        auto car = getRefScenario()->cars.at(i);
        IntelligentDriverModel::advanceStep(car);
    }
}

void OpenMPAlgorithm::advanceTrafficLights() {
#pragma omp single
    for (auto pair : getRefScenario()->junctions) {
        pair.second->updateSignals();
    }
}

void OpenMPAlgorithm::sortLanesAndCalculateAcceleration() {
#pragma omp for
    for (long i = 0; i < getRefScenario()->lanes.size(); i++) {
        auto lane = getRefScenario()->lanes.at(i);
        if (!lane->isSorted) {
            std::sort(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), TrafficObject::Cmp());
            lane->isSorted = true;
        }
        for (std::size_t i = 0; i < lane->mTrafficObjects.size(); i++) {
            auto car = lane->mTrafficObjects.at(i);
            auto leadingObject = (i < lane->mTrafficObjects.size() - 1) ? lane->mTrafficObjects.at(i + 1) : nullptr;

            car->calcSameLaneAcceleration(leadingObject);
        }
    }
}

void OpenMPAlgorithm::advance(size_t steps) {

#pragma omp parallel
    for (int i = 0; i < steps; i++) {
        sortLanesAndCalculateAcceleration();
        calculateCarChanges();
        advanceCars();
        advanceTrafficLights();
    }
}