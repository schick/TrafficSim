#include <fstream>
#include <iostream>

#include "Scenario.h"

#ifdef VISUALIZATION_ENABLED
#include "Visualization.h"
#endif

int main(int argc, char* argv[])
{
    json input, loesung;

    // read input file
#ifdef USE_CIN
    std::cin >> input;
#else

    std::string fn(argv[1]);
    std::ifstream json_file(fn);
    try {
        json_file >> input;
    } catch(const std::exception &e) {
        std::cerr << "Failed to parse JSON.\n" << e.what() << std::endl;
        return 1;
    }

    // read loesung
    std::ifstream json_file_out(fn + ".sol");
    if (json_file_out.good()) {
        try {
            json_file_out >> loesung;
        } catch (const std::exception &e) {
            std::cerr << "Failed to parse JSON.\n";
        }
    }
#endif

    Scenario scenario;
    scenario.parse(input);

    OkesExampleAdvanceAlgorithm advancer(&scenario);

#ifdef VISUALIZATION_ENABLED
    Visualization visualization(&scenario);
    std::string video_fn("output.avi");
    visualization.setVideoPath(video_fn, 1);
    visualization.render_image();
#endif

    for(int i=0; i < input["time_steps"]; i++) {
        advancer.advance();

#ifdef VISUALIZATION_ENABLED
        visualization.render_image();
#endif
    }

#ifdef VISUALIZATION_ENABLED
    visualization.render_image();
    visualization.close();
#endif

    json output = scenario.toJson();
    std::cout << output.dump() << "\n";

#ifndef USE_CIN
    if (json_file_out.good()) {
        std::cout << loesung.dump() << "\n";
        assert(output == loesung);
    }
#endif

    return 0;
}
