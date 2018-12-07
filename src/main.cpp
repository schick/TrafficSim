#include <fstream>
#include <iostream>

#include "Scenario.h"

#define VISUALIZATION

#ifdef VISUALIZATION
#include "Visualization.h"
#endif

int main(int argc, char* argv[])
{
    json input, loesung;
    std::string fn(argv[1]);

    // read input file
    std::ifstream json_file(fn);
    try {
        json_file >> input;
    } catch(std::exception e) {
        std::cerr << "Failed to parse JSON.\n";
        return 1;
    }

    // read loesung
    std::ifstream json_file_out(fn + ".sol");
    try {
        json_file_out >> loesung;
    } catch(std::exception e) {
        std::cerr << "Failed to parse JSON.\n";
        return 1;
    }

    Scenario scenario;
    scenario.parse(input);

    OkesExampleAdvanceAlgorithm advancer(&scenario);

#ifdef VISUALIZATION
    Visualization visualization(&scenario);
    std::string video_fn("output.avi");
    visualization.open(video_fn, 1);
    visualization.render_image();
#endif

    for(int i=0; i < input["time_steps"]; i++) {
        advancer.advance();

#ifdef VISUALIZATION
        visualization.render_image();
#endif

    }
#ifdef VISUALIZATION
    visualization.render_image();
    visualization.close();
#endif


    json output = scenario.toJson();

    std::cout << output.dump() << "\n";
    std::cout << loesung.dump() << "\n";

    assert(output == loesung);

    return 0;
}
