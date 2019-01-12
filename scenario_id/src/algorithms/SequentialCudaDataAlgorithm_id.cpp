//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialCudaDataAlgorithm_id.h"

void SequentialCudaDataAlgorithm_id::advance(size_t steps) {
    std::vector<Car_id::AdvanceData> changes(cudaScenario.getNumCars());
    Scenario_id &scenario_id = *dynamic_cast<Scenario_id*>(getScenario().get());

    {
        for (int i = 0; i < steps; i++) {
            #pragma omp parallel for
            for (long c_i = 0; c_i < cudaScenario.getNumCars(); c_i++) {
                changes[c_i] = wrapper.nextStep(*cudaScenario.getCar(c_i));
            }

            #pragma omp parallel for
            for (long c_i = 0; c_i < cudaScenario.getNumCars(); c_i++) {
                wrapper.advanceStep(*cudaScenario.getCar(changes[c_i].car), changes[c_i]);
            }

            #pragma omp parallel for
            for (long c_i = 0; c_i < cudaScenario.getNumJunctions(); c_i++) {
                wrapper.updateSignals(*cudaScenario.getJunction(c_i));
            }
        }
    }
}
