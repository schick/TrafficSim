//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"


void SequentialAlgorithm::calculateCarChanges() {
    for (std::unique_ptr<Road> &r : getRefScenario()->roads) {
        for (auto &l : r.get()->lanes) {
            for (auto it = l->mTrafficObjects.begin(); it != l->mTrafficObjects.end(); ++it) {
                //Iterate over cars of lane. neighbors are it+1 and it-1.
                Lane::NeighboringObjects neighbors;

                if (it != l->mTrafficObjects.begin())
                    neighbors.back = *(it - 1);

                if (l->mTrafficObjects.end() == it) {
                    if (it != l->mTrafficObjects.end())
                        neighbors.front = *it;
                }
                else {
                    if (it + 1 != l->mTrafficObjects.end())
                        neighbors.front = *(it + 1);
                }
                idm.nextStep(dynamic_cast<Car*>(*it), neighbors);
            }
        }
    }
};

void SequentialAlgorithm::advanceCars() {
    for (std::unique_ptr<Car> &car : getRefScenario()->cars) {
        idm.advanceStep(car.get());
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
