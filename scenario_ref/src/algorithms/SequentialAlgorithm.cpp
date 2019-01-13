//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"


void SequentialAlgorithm::calculateCarChanges() {
    for (std::shared_ptr<Road> &r : getRefScenario()->roads) {
        for (auto &l : r.get()->lanes) {
            for (std::size_t i = 0; i < l->mTrafficObjects.size(); i++) {
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


void SequentialAlgorithm::advance(size_t steps) {
    for (int i = 0; i < steps; i++) {
        sortLanes();
        calculateCarChanges();
        advanceCars();
        advanceTrafficLights();
    }
}
