//
// Created by maxi on 1/20/19.
//

#include "optimization/model/SignalLayout.h"
#include <random>
#include <algorithm>

// Helper methods
inline unsigned long range_random(size_t min, size_t max) {
    std::random_device rd1;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd1());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<size_t> uni(min,max);
    return uni(rng);
}

inline unsigned long range_random(size_t max) {
    return range_random(0, max);
}

SignalLayout::SignalLayout(OptimizeScenario &scenario) {
    for (Junction &junction : scenario.junctions)
        createRandomSignal(junction);
}

void SignalLayout::populate(OptimizeScenario &scenario) {
    for (Junction &junction : scenario.junctions)
        junction.signals = signalsMap.at(junction.id);

    scenario.initJunctions();
}

void SignalLayout::createRandomSignal(Junction &junction) {
    uint64_t junctionId = junction.id;
    std::vector<Junction::Direction> possibleDirections = junction.getPossibleDirections();
    //is deprecated: std::random_shuffle(possibleDirections.begin(), possibleDirections.end()) 
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(possibleDirections.begin(), possibleDirections.end(), g);

    std::vector<Junction::Signal> signalsVector;

    for (Junction::Direction &direction : possibleDirections)
        signalsVector.emplace_back(range_random(5, 10), direction);

    signalsMap.insert({junctionId, signalsVector});
}


