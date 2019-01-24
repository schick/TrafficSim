//
// Created by maxi on 1/20/19.
//

#include "optimization/model/SignalLayout.h"
#include <random>
#include <algorithm>
#include <AdvanceAlgorithm.h>

// Helper methods
inline unsigned long range_random(size_t min, size_t max) {
    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<size_t> uni(min,max);
    return uni(rng);
}

inline unsigned long range_random(size_t max) {
    return range_random(0, max);
}

inline void shuffleDirections(std::vector<Junction::Direction> &possibleDirections) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(possibleDirections.begin(), possibleDirections.end(), rng);
}

// Constructor
SignalLayout::SignalLayout(std::string algorithm, nlohmann::json scenarioData) {

    // Instantiate advancer
    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
    if (advancer == nullptr) {
        throw std::runtime_error("Algorithm not found: " + algorithm);
    }

    // Getting pointer to scenario
    auto *scenario = dynamic_cast<OptimizeScenario *>(advancer->getScenario().get());
    if (scenario == nullptr) {
        throw std::runtime_error("Algorithm '" + algorithm + "' with wrong scenario type for 'RandomOptimizer'");
    }

    // Creating random signals
    for (Junction &junction : scenario->junctions)
        createRandomSignal(junction);

    // Populating scenario
    populate(*scenario);

    // Run advancer and save traveled distance and json
    advancer->advance(scenarioData["time_steps"]);
    travelledDistance = scenario->getTravelledDistance();
}

// Merge Constructor
SignalLayout::SignalLayout(SignalLayout firstParent, SignalLayout secondParent, std::string algorithm, nlohmann::json scenarioData) {

    // Instantiate advancer
    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
    if (advancer == nullptr) {
        throw std::runtime_error("Algorithm not found: " + algorithm);
    }

    // Getting pointer to scenario
    auto *scenario = dynamic_cast<OptimizeScenario *>(advancer->getScenario().get());
    if (scenario == nullptr) {
        throw std::runtime_error("Algorithm '" + algorithm + "' with wrong scenario type for 'RandomOptimizer'");
    }

    for (Junction &junction : scenario->junctions) {

        // Draw random value
        unsigned long random = range_random(1, 100);

        if (random <= 45) {
            signalsMap.insert({junction.id, firstParent.signalsMap.at(junction.id)});
        } else if (random <= 90) {
            signalsMap.insert({junction.id, secondParent.signalsMap.at(junction.id)});
        } else {
            createRandomSignal(junction);
        }
    }

    // Populating scenario
    populate(*scenario);

    // Run advancer and save traveled distance and json
    advancer->advance(scenarioData["time_steps"]);
    travelledDistance = scenario->getTravelledDistance();
}

// Class Helper
void SignalLayout::populate(OptimizeScenario &scenario) {
    for (Junction &junction : scenario.junctions)
        junction.signals = signalsMap.at(junction.id);

    scenario.initJunctions();
}

void SignalLayout::createRandomSignal(Junction &junction) {
    uint64_t junctionId = junction.id;
    std::vector<Junction::Direction> possibleDirections = junction.getPossibleDirections();
    shuffleDirections(possibleDirections);

    std::vector<Junction::Signal> signalsVector;
    signalsVector.reserve(possibleDirections.size());

    for (Junction::Direction &direction : possibleDirections)
        signalsVector.emplace_back(range_random(5, 10), direction);

    signalsMap.insert({junctionId, signalsVector});
}

// To JSON
nlohmann::json SignalLayout::toJson() {
    json output;
    for (auto pair : signalsMap) {
        json out_junction;

        out_junction["id"] = pair.first;

        for (Junction::Signal &signal : pair.second) {
            json out_signal;

            out_signal["dir"] = static_cast<int>(signal.direction);
            out_signal["time"] = signal.duration;

            out_junction["signals"].push_back(out_signal);
        }

        output["junctions"].push_back(out_junction);
    }
    return output;
}


