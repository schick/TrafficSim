//
// Created by oke on 16.12.18.
//

#include "register_algorithms.h"
#include "model/Scenario.h"
#include <chrono>
#include "util/json.hpp"
#include <fstream>

int main() {
    constexpr int steps = 1000;

    std::vector<std::string> files = {
            "tests/own_tests/16x16.json",
            "tests/own_tests/32x32.json",
            "tests/own_tests/64x64.json",
            "tests/own_tests/128_5_20695_103475.json",
            "tests/own_tests/256_5_83.751_418.755.json"
    };

    std::vector<std::string> algorithms = AdvanceAlgorithm::getAlgorithms();
    // std::swap(*std::find(algorithms.begin(), algorithms.end(), "SequentialAlgorithm"), algorithms[0]);
    std::vector<long> durations_total;
    std::vector<long> durations;
    for (std::string &fn : files) {
        for(std::string &a_name : algorithms) {
            printf("Benchmark %s\n", a_name.c_str());
            printf("    - Load scenario %s\n", fn.c_str());

            json input;
            std::ifstream json_file(fn);
            json_file >> input;
            auto before_parsing = std::chrono::system_clock::now();
            std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiate(a_name, input);

            printf("    - execute test for %d steps\n", steps);
            auto start = std::chrono::system_clock::now();
            advancer->advance(steps);
            auto end = std::chrono::system_clock::now();
            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            durations.push_back(milliseconds);
            auto milliseconds_total = std::chrono::duration_cast<std::chrono::milliseconds>(end - before_parsing).count();
            durations_total.push_back(milliseconds_total);
            printf("Done in %ldms (%ldms)\n\n", milliseconds, milliseconds_total);
        }
    }

    printf("Speedups:\n");
    std::vector<double> means(algorithms.size());
    for(int a_idx=0; a_idx<algorithms.size(); a_idx++) {
        for(int f_idx=0; f_idx<files.size(); f_idx++) {
            means[a_idx] = (double) durations[f_idx * algorithms.size()] / (double) durations[a_idx + f_idx * algorithms.size()];
        }
    }

    printf("\n");

    for(int a_idx=1; a_idx<algorithms.size(); a_idx++) {
        printf("%s: %.2f\n", algorithms[a_idx].c_str(), means[a_idx]);
    }

    printf("Speedups with Parsing:\n");
    std::vector<double> means_total(algorithms.size());
    for(int a_idx=0; a_idx<algorithms.size(); a_idx++) {
        for(int f_idx=0; f_idx<files.size(); f_idx++) {
            means_total[a_idx] = (double) durations_total[f_idx * algorithms.size()] / (double) durations_total[a_idx + f_idx * algorithms.size()];
        }
    }
    for(int a_idx=1; a_idx<algorithms.size(); a_idx++) {
        printf("%s: %.2f\n", algorithms[a_idx].c_str(), means_total[a_idx]);
    }

}