#include <fstream>
#include <iostream>

#include "util/SimpleArgumentParser.h"
#include "model/Scenario.h"
#include "AdvanceAlgorithm.h"

#include "register_algorithms.h"
#include "util/json.hpp"

#ifdef VISUALIZATION_ENABLED
#include "BaseVisualizationEngine.h"
#include "Visualization_id.h"
#endif
int main(int argc, char* argv[])
{
    json input;

    SimpleArgumentParser p;
    p.add_kw_argument("algorithm", "OpenMPAlgorithm");

    // read input file
#ifdef USE_CIN
    std::cin >> input;
    p.load(argc, argv);
#else
    p.add_argument("file_name");
    p.load(argc, argv);

    std::string file_name(p["file_name"]);
    std::ifstream json_file(file_name);
    try {
        json_file >> input;
    } catch(const std::exception &e) {
        std::cerr << "Failed to parse JSON.\n" << e.what() << std::endl;
        return 1;
    }

    // read loesung
#ifdef DEBUG_MSGS
    json loesung;
    std::ifstream json_file_out(file_name + ".sol");
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
    std::cout << output.dump() << "\n";

#ifndef USE_CIN
#ifdef DEBUG_MSGS
    if (json_file_out.good()) {
        std::cout << loesung.dump() << "\n";

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
#endif

#ifdef DEBUG_MSGS
    std::cout << "Time: " << elapsed.count() << "ms" << std::endl;
#endif


    return 0;
}
