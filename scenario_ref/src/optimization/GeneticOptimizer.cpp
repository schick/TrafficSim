//
// Created by maxi on 1/23/19.
//

#include "optimization/GeneticOptimizer.h"
#include <algorithm>
#include <optimization/model/SignalLayout.h>

nlohmann::json GeneticOptimizer::optimize() {

    constexpr size_t POPULATIONS_SIZE = 120;

    std::vector<SignalLayout> population;
    population.reserve(POPULATIONS_SIZE);

    for (size_t i = 0; i < POPULATIONS_SIZE; i++)
        population.emplace_back(algorithm, scenarioData);

    std::sort(population.begin(), population.end(), SignalLayout::Cmp());

    iterations += 120;

    while (population.front().getTravelledDistance() < minTravelLength) {

        std::vector<SignalLayout> newPopulation;
        newPopulation.reserve(POPULATIONS_SIZE);

        // Merge the 16 best layouts with each other
        // Exactly 120 iterations
        for (size_t i = 0; i < 16; i++) {
            for (size_t j = (i + 1); j < 16; j++) {
                newPopulation.emplace_back(population[i], population[j], algorithm, scenarioData);
            }
        }

        std::sort(newPopulation.begin(), newPopulation.end(), SignalLayout::Cmp());

        population = newPopulation;

        iterations += 120;
    }

    return population.front().toJson();

}