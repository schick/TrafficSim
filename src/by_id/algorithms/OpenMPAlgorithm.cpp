//
// Created by oke on 15.12.18.
//

#include "algorithms/OpenMPAlgorithm.h"

void OpenMPAlgorithm::advance(size_t steps) {

    auto &cars = getScenario()->cars;
    auto &lights = getScenario()->junctions;
    std::vector<Car::AdvanceData> changes(cars.size());
    #pragma omp parallel shared(cars, changes)
    {
        for (int i = 0; i < steps; i++) {
            #pragma omp for
            for (int i = 0; i < cars.size(); i++) {
                changes[i] = cars[i].nextStep(*getScenario());
            }

            #pragma omp for
            for (int i = 0; i < changes.size(); i++) {
                getScenario()->cars[changes[i].car].advanceStep(*getScenario(), changes[i]);
            }

            #pragma omp for
            for (int i = 0; i < lights.size(); i++) {
                lights[i].updateSignals(*getScenario());
            }
        }
    }
}
