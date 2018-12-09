#include <fstream>
#include <iostream>

#include "Scenario.h"

//#ifdef VISUALIZATION_ENABLED
#include "Visualization.h"
//#endif

int main(int argc, char* argv[])
{
    json input, loesung;
/*#ifdef USE_CIN
    std::cin >> input;
#else*/
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
//#endif

    Scenario scenario;
    scenario.parse(input);

    OkesExampleAdvanceAlgorithm advancer(&scenario);

//#ifdef VISUALIZATION_ENABLED
    Visualization visualization(&scenario);
    std::string video_fn("output.avi");
    visualization.open(video_fn, 1);
    visualization.render_image();
//#endif

    for(int i=0; i < input["time_steps"]; i++) {
        advancer.advance();

//#ifdef VISUALIZATION_ENABLED
        visualization.render_image();
//#endif

    }
//#ifdef VISUALIZATION_ENABLED
    visualization.render_image();
    visualization.close();
//#endif


    json output = scenario.toJson();

    std::cout << output.dump() << "\n";

#ifndef USE_CIN
    std::cout << loesung.dump() << "\n";
    assert(output == loesung);
#endif

    return 0;
}
