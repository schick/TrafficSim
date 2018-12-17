//
// Created by oke on 16.12.18.
//

#include "algorithms/CudaAlgorithm.h"
#include "algorithms/cuda_algorithms.cuh"


void CudaAlgorithm::advance(size_t steps) {
    next(*getScenario());
    auto &cars = getScenario()->cars;
    auto &lights = getScenario()->junctions;
    std::vector<Car::AdvanceData> changes(cars.size());
    {
        for (int i = 0; i < steps; i++) {
            for (int i = 0; i < cars.size(); i++) {
                changes[i] = cars[i].nextStep(*getScenario());
            }

            for (int i = 0; i < changes.size(); i++) {
                getScenario()->cars[changes[i].car].advanceStep(*getScenario(), changes[i]);
            }

            for (int i = 0; i < lights.size(); i++) {
                lights[i].updateSignals(*getScenario());
            }
        }
    }
}