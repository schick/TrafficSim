//
// Created by oke on 15.12.18.
//

#include "algorithms/OpenMPAlgorithm_id.h"
/*
void OpenMPAlgorithm_id::advance(size_t steps) {

    auto &cars = getIDScenario()->cars;
    auto &lights = getIDScenario()->junctions;
    std::vector<Car_id::AdvanceData> changes(cars.size());
    #pragma omp parallel shared(cars, changes)
    {
        for (int i = 0; i < steps; i++) {
            #pragma omp for
            for (int i = 0; i < cars.size(); i++) {
                changes[i] = cars[i].nextStep(*getIDScenario());
            }

            #pragma omp for
            for (int i = 0; i < changes.size(); i++) {
                getIDScenario()->cars[changes[i].car].advanceStep(*getIDScenario(), changes[i]);
            }

            #pragma omp for
            for (int i = 0; i < lights.size(); i++) {
                lights[i].updateSignals(*getIDScenario());
            }
        }
    }
}
*/