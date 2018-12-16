#include <fstream>
#include <iostream>

#include "util/SimpleArgumentParser.h"
#include "Scenario.h"
#include "algorithms/AdvanceAlgorithm.h"

#ifdef VISUALIZATION_ENABLED
#include "util/Visualization.h"
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

    Scenario scenario;
    scenario.parse(input);

    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiate(p["algorithm"], &scenario);
    if (advancer == nullptr) {
        printf("Algorithm not found.");
        exit(-1);
    }
#ifdef VISUALIZATION_ENABLED
    Visualization visualization(&scenario);
    std::string video_fn("output.avi");
    visualization.setVideoPath(video_fn, 1);
    visualization.render_image();
#endif

#ifdef DEBUG_MSGS
    auto start = std::chrono::system_clock::now();
#endif

#ifdef VISUALIZATION_ENABLED
    for(int i=0; i < input["time_steps"]; i++) {
        advancer->advance(1);
        visualization.render_image();
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
    visualization.render_image();
    visualization.close();
#endif

    json output = scenario.toJson();
    std::cout << output.dump() << "\n";

#ifndef USE_CIN
#ifdef DEBUG_MSGS
    if (json_file_out.good()) {
        std::cout << loesung.dump() << "\n";
        if(output != loesung) {
            printf("Lösungen nicht gleich...\n");
        }
    }
#endif
#endif

#ifdef DEBUG_MSGS
    std::cout << "Time: " << elapsed.count() << "ms" << std::endl;
#endif


    return 0;
}
