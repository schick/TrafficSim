//
// Created by maxi on 1/18/19.
//

#include "optimization/DistributionOptimizer.h"
#include "optimization/model/OptimizeScenario.h"
#include "AdvanceAlgorithm.h"
#include <memory>
#include <stdexcept>
#include <random>
#include <optimization/model/SignalLayout.h>

inline unsigned int range_random(size_t min, size_t max) {
    return rand() % (max - min) + min;
}

inline unsigned int range_random(size_t max) {
    return range_random(0, max);
}

long seed = std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine generator(seed);


void _randomInitialization(OptimizeScenario &scenario, std::vector<std::array<double, 4>> &incoming_counts,
        size_t mean_max_duration_min=9, size_t mean_max_duration_max=15, size_t normal_std=1) {

    for(auto &junction : scenario.junctions) {
        junction.signals.resize(0);
        std::vector signal_idxs = {0, 1, 2, 3};
        double *max_metric = std::max_element(incoming_counts[junction.id].begin(), incoming_counts[junction.id].end());
        size_t mean_max_duration = range_random(mean_max_duration_min, mean_max_duration_max);

        if(*max_metric > 1e-4) {
            for (size_t idx = 0; idx < 4; idx += 1) {
                size_t i = range_random(signal_idxs.size());
                size_t signal_idx = signal_idxs[i];
                signal_idxs.erase(signal_idxs.begin() + i);
                std::normal_distribution<double> distribution(
                        incoming_counts[junction.id][signal_idx] / *max_metric * mean_max_duration, normal_std);
                auto duration = (size_t) std::abs(std::round(distribution(generator)));
                if (duration < 5 && duration > 2) duration = 5;
                if (duration <= 2) continue;
                junction.signals.emplace_back(duration, (Junction::Direction) signal_idx);
                if (duration < 5) throw std::runtime_error("Duration is too small. " + std::to_string(duration));
            }
        } else {
            size_t max_idx = max_metric - incoming_counts[junction.id].begin();
            junction.signals.emplace_back(5, (Junction::Direction ) std::min((size_t) 3, max_idx));
        }
        if(junction.signals.size() == 0) {
            size_t max_idx = 0;
            for(size_t i=0; i < 4; i++) if (junction.incoming[i] != nullptr) max_idx = i;
            junction.signals.emplace_back(5, (Junction::Direction ) max_idx);
        }
        assert(!junction.signals.empty());
    }
    scenario.initJunctions();
}

void zeroInitialization(OptimizeScenario &scenario) {
    for(auto &junction : scenario.junctions) {
        junction.signals.clear();
    }
    scenario.initJunctions();
}

std::vector<std::array<double, 4>> DistributionOptimizer::initialSimulation() {
    std::shared_ptr<BaseScenario> last_scenario;

    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
    if (advancer == nullptr) throw std::runtime_error("Algorithm not found: " + algorithm);

    auto *scenario = dynamic_cast<OptimizeScenario *>(advancer->getScenario().get());
    if (scenario == nullptr) throw std::runtime_error("Algorithm '" + algorithm + "' with wrong scenario type for 'RandomOptimizer'");

    zeroInitialization(*scenario);
    advancer->advance(scenarioData["time_steps"]);

#ifdef DEBUG_MSGS
    printf("Max distance: %.2f\n", scenario->getTravelledDistance());
#endif

    std::vector<std::array<double, 4>> incoming_counts;
    incoming_counts.reserve(scenario->junctions.size());
    for(Junction &j : scenario->junctions) incoming_counts.emplace_back(j.incoming_counter);
    return incoming_counts;
}


bool DistributionOptimizer::IsDone() {
    return !validResults.empty();
}

void DistributionOptimizer::randomTestsUntilDone(std::vector<std::array<double, 4>> &incoming_counts) {

    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiateOptimization(algorithm, scenarioData);
    if (advancer == nullptr) throw std::runtime_error("Algorithm not found: " + algorithm);

    auto *scenario = dynamic_cast<OptimizeScenario *>(advancer->getScenario().get());
    if (scenario == nullptr) throw std::runtime_error("Algorithm '" + algorithm + "' with wrong scenario type for 'RandomOptimizer'");

    int idx = 0;
    while (!IsDone()) {
        idx++;
        scenario->reset();
        if(idx % 2 == 1) {
            SignalLayout signalLayout(*scenario);
            signalLayout.populate(*scenario);
        } else {
            _randomInitialization(*scenario, incoming_counts, 5, 15, 1);
        }

        advancer->advance(scenarioData["time_steps"]);

        double total_distance = scenario->getTravelledDistance();

#ifdef DEBUG_MSGS
        printf("Distance: %.2f / %.2f\n", total_distance, minTravelLength);
#endif
        if (total_distance > minTravelLength) {
            std::scoped_lock lock(validResultsMutex);
            validResults.push_back(scenario->toJson());
        }
    }
}



nlohmann::json DistributionOptimizer::optimize() {

    std::vector<std::array<double, 4>> incoming_counts = initialSimulation();

    randomTestsUntilDone(incoming_counts);

    return validResults.front();
}
