//
// Created by oke on 15.12.18.
//

#include "algorithms/OpenMPAlgorithm.h"


void OpenMPAlgorithm::calculateCarChanges() {
    for (std::shared_ptr<Road> &r : getRefScenario()->roads) {
        for (auto &l : r.get()->lanes) {
            for (auto it = l->mTrafficObjects.begin(); it != l->mTrafficObjects.end(); ++it) {
                //Iterate over cars of lane. neighbors are it+1 and it-1.
                Lane::NeighboringObjects neighbors;

                //set preceding car for all cars except the first one
                if (it != l->mTrafficObjects.begin())
                    neighbors.back = *(it - 1);

                //set next car for all cars except the last one
                if (it != l->mTrafficObjects.end())
                    neighbors.front = *(it + 1);

                (*it)->nextStep(neighbors);
            }
        }
    }
};

void OpenMPAlgorithm::advanceCars() {
    for (std::shared_ptr<Car> &car : getRefScenario()->cars) {
        IntelligentDriverModel::advanceStep(car.get());
    }
}

void OpenMPAlgorithm::advanceTrafficLights() {
    for (std::shared_ptr<Junction> &j : getRefScenario()->junctions) {
        j->updateSignals();
    }
}

void OpenMPAlgorithm::sortLanes() {
    for (auto &lane : getRefScenario()->lanes) {
        //auto trafficObjects = lane->mTrafficObjects;
        std::sort(lane->mTrafficObjects.begin(), lane->mTrafficObjects.end(), TrafficObject::Cmp());
    }
}


void OpenMPAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        sortLanes();
        calculateCarChanges();
        advanceCars();
        advanceTrafficLights();
    }
}