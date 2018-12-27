//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"


void SequentialAlgorithm::calculateCarChanges() {
    for (std::unique_ptr<Road> &r : getRefScenario()->roads) {
        for (auto &l : r.get()->lanes) {
            for (auto c : l->mTrafficObjects) {
                idm.nextStep(dynamic_cast<Car*>(c));
            }
        }
    }
};

void SequentialAlgorithm::advanceCars() {
    for (std::unique_ptr<Car> &car : getRefScenario()->cars) {

        idm.advanceStep(car.get());
        //car->new_lane_offset = 0;
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
        calculateCarChanges();
        advanceCars();
        advanceTrafficLights();
    }
}
