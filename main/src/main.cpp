

#include <fstream>
#include <iostream>

#include "TrafficSim.h"

#include "util/json.hpp"
#include "util/SimpleArgumentParser.h"

#ifdef VISUALIZATION_ENABLED
#include "BaseVisualizationEngine.h"
#include "Visualization_id.h"
#endif

int main(int argc, char* argv[]) {

    SimpleArgumentParser p;
    nlohmann::json input;

    // set default algorithm
#ifdef WITH_CUDA
    p.add_kw_argument("algorithm", "CudaAlgorithm");
#else
    p.add_kw_argument("algorithm", "OpenMPAlgorithm");
#endif

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
#endif

    if (input.find("optimize_signals") != input.end() && input["optimize_signals"] == true) {
        trafficSim::optimize(input, p);
    } else {
        trafficSim::calculate(input, p);
    }


    return 0;
}
