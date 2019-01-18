//
// Created by maxi on 1/17/19.
//

#include "TrafficSim.h"
#include "AdvanceAlgorithm.h"

#include <iostream>
#include <fstream>
#include <chrono>

void trafficSim::optimize(nlohmann::json &input, SimpleArgumentParser &p) {

}

void trafficSim::calculate(nlohmann::json &input, SimpleArgumentParser &p) {

// read input file
#ifndef USE_CIN
#ifdef DEBUG_MSGS
    // read loesung
    json loesung;
    std::ifstream json_file_out(p["file_name"] + ".sol");
    if (json_file_out.good()) {
        try {
            json_file_out >> loesung;
        } catch (const std::exception &e) {
            std::cerr << "Failed to parse JSON.\n";
        }
    }
#endif
#endif

    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiate(p["algorithm"], input);
    if (advancer == nullptr) {
        printf("Algorithm not found.");
        exit(-1);
    }

#ifdef VISUALIZATION_ENABLED
    std::shared_ptr<BaseVisualizationEngine> visualization = advancer->createVisualizationEngine(advancer->getScenario());
    std::string video_fn("output.avi");
    visualization->setVideoPath(video_fn, 1);
    visualization->render_image();
#endif

#ifdef DEBUG_MSGS
    auto start = std::chrono::system_clock::now();
#endif

#ifdef VISUALIZATION_ENABLED
    for(int i=0; i < input["time_steps"]; i++) {
        advancer->advance(1);
        visualization->render_image();
    }
#else
    advancer->advance(input["time_steps"]);
#endif

#ifdef DEBUG_MSGS
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start);
#endif

#ifdef VISUALIZATION_ENABLED
    visualization->render_image();
    visualization->close();
#endif

    json output = advancer->getScenario()->toJson();

#ifdef USE_CIN
    std::cout << output.dump() << "\n";
#endif

#ifdef RUN_WITH_TESTS
    std::cout << "Output:   " << output.dump() << "\n";
        if (json_file_out.good()) {
            std::cout << "Expected: " << loesung.dump() << "\n";

            for (auto &car_json : output["cars"]) {
                bool found_car = false;
                for (auto &cmp_car_json : loesung["cars"]) {
                    if (cmp_car_json["id"] == car_json["id"] &&
                        cmp_car_json["to"] == car_json["to"] &&
                        cmp_car_json["from"] == car_json["from"] &&
                        abs((double)cmp_car_json["position"] - (double)car_json["position"]) <= 1e-7 &&
                        cmp_car_json["lane"] == car_json["lane"]) {
                        found_car = true;
                        break;
                    }
                }
                if (!found_car) fprintf(stderr, "Car(%d) is not correct.\n", (int) car_json["id"]);
            }
        }
#endif

#ifdef DEBUG_MSGS
    std::cout << "Time: " << elapsed.count() << "ms" << std::endl;
#endif
}

