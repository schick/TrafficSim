//
// Created by oke on 15.12.18.
//

#include "algorithms/OpenMPAlgorithm.h"

void OpenMPAlgorithm::advance(size_t steps) {

    auto &cars = getRefScenario()->cars;
    auto &lights = getRefScenario()->junctions;
    auto &lanes = getRefScenario()->lanes;
    //std::vector<Car::AdvanceData> changes(cars.size());
    //#pragma omp parallel shared(cars, changes)
    {
        /*for (int i = 0; i < steps; i++) {
            
            
            #pragma omp parallel for //shared(lanes)
            for (int i = 0; i < lanes.size(); i++) {
                //auto trafficObjects = lane->mTrafficObjects;
                std::sort(lanes[i]->mTrafficObjects.begin(), lanes[i]->mTrafficObjects.end(), TrafficObject::Cmp());
            }

            #pragma omp parallel for //shared(changes, cars)
            for (int i = 0; i < cars.size(); i++) {
                changes[i] = idm.nextStep(cars[i].get());
            }

            #pragma omp parallel for //shared(changes)
            for (int i = 0; i < changes.size(); i++) {
                idm.advanceStep(changes[i], changes[i].car);
            }

            #pragma omp parallel for //shared(lights)
            for (int i = 0; i < lights.size(); i++) {
                lights[i]->updateSignals();
            }
        }*/
    }
}
