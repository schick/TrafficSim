//
// Created by oke on 22.12.18.
//

#include <fstream>
#include <iostream>
#include "util/json.hpp"
#include "AdvanceAlgorithm.h"
#include "register_algorithms.h"


int main() {
    nlohmann::json input;
    std::string file_name("../tests/44-tiny_400timestep.json");
    std::ifstream json_file(file_name);
    try {
        json_file >> input;
    } catch(const std::exception &e) {
        std::cerr << "Failed to parse JSON.\n" << e.what() << std::endl;
        return 1;
    }

    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiate("SequentialAlgorithm", input);
    std::shared_ptr<AdvanceAlgorithm> advancer2 = AdvanceAlgorithm::instantiate("SequentialCudaDataAlgorithm_id", input);


    std::shared_ptr<BaseVisualizationEngine> visualization1 = advancer->createVisualizationEngine(advancer->getScenario());
    std::string video_fn("/home/oke/Desktop/testimages/output");
    visualization1->setImageBasePath(video_fn);
    visualization1->render_image();


    std::shared_ptr<BaseVisualizationEngine> visualization2 = advancer2->createVisualizationEngine(advancer2->getScenario());
    std::string video_fn2("/home/oke/Desktop/testimages/output_new");
    visualization2->setImageBasePath(video_fn2);
    visualization2->render_image();



    Scenario_id* scenarioid = dynamic_cast<Scenario_id*>(advancer2->getScenario().get());
    Scenario* scenario = dynamic_cast<Scenario*>(advancer->getScenario().get());

    visualization1->render_image();
    visualization2->render_image();
    for(int i=0; i < input["time_steps"]; i++) {
        printf("Step: %d:\n", i);
        if (i == 79) {
            printf("bla\n");
        }
        advancer->advance(1);
        advancer2->advance(1);
        std::cout << "re-json: " << scenario->toJson() << std::endl;
        std::cout << "ID-json: " << scenarioid->toJson() << std::endl;
        std::cout << "Exact: " << (scenario->toJson() == scenarioid->toJson()) << std::endl;
        visualization2->render_image();
        visualization1->render_image();


        for(auto &car : scenario->cars) {
            for (auto &car2 : scenarioid->cars) {
                if (car->id == car2.id) {
                    Lane_id &l2 = scenarioid->lanes.at(car2.lane);
                    Road_id &r2 = scenarioid->roads.at(l2.road);
                    if(car->getLane()->road->from->id != scenarioid->junction_working_to_original_ids[r2.from] ||
                       car->getLane()->road->to->id != scenarioid->junction_working_to_original_ids[r2.to]) {
                        printf("Car(%d) is at wrong road in step %d\n", car->id , i);
                        exit(0);
                    }
                    if(car->getLane()->lane != l2.lane_num) {
                        printf("Car(%d) is at wrong lane in step %d\n", car->id , i);
                        exit(0);
                    }

                    if(fabs(car->getPosition() - car2.x) > 1e-4) {
                        printf("Car(%d) is at wrong position in step %d\n", car->id , i);
                        exit(0);
                    }
                }
            }
        }
    }
    visualization1->render_image();
    visualization2->render_image();

    return 0;
}