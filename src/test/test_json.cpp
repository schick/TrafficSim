
#include <fstream>
#include <gtest/gtest.h>

#include "util/json.hpp"
#include "Scenario.h"
#include "algorithms/SequentialAlgorithm.h"
#include "register_algorithms.h"


using json = nlohmann::json;

void test_file(std::string algorithm, std::string fn, double genauigkeit) {

    json input, loesung;

    // read input file
    std::ifstream json_file(fn);
    EXPECT_NO_THROW(json_file >> input);
    // read loesung
    std::ifstream json_file_out(fn + ".sol");
    EXPECT_NO_THROW(json_file_out >> loesung);

    std::shared_ptr<AdvanceAlgorithm> advancer = AdvanceAlgorithm::instantiate(algorithm, input);
    advancer->advance(input["time_steps"]);

    json output = advancer->getScenario()->toJson();

    ASSERT_EQ(loesung["cars"].size(), output["cars"].size());
    bool error = false;
    for(auto &car : output["cars"]) {
        for (auto &lcar : loesung["cars"]) {
            if (car["id"] == lcar["id"]) {
                if (car["from"] != lcar["from"] || car["to"] != car["to"]) {
                    printf("Car(%d) is at wrong road\n", (int) car["id"]);
                    error = true;
                    continue;
                }

                if(car["lane"] != lcar["lane"]) {
                    printf("Car(%d) is at wrong lane\n", (int) car["id"]);
                    error = true;
                    continue;
                }

                if(fabs((double) car["position"] - (double) lcar["position"]) > genauigkeit) {
                    printf("Car(%d) is at wrong position (error: %f)\n", (int) car["id"], fabs((double) car["position"] - (double) lcar["position"]));
                    error = true;
                }
            }
        }
    }
    ASSERT_FALSE(error);
}

#define JSON_TEST_PATH std::string("tests/")

#define _CREATE_TEST(NAME, PATH, ALGO, ACCURACY) TEST(test##ALGO##ACCURACY, NAME) {\
    test_file(STR(ALGO), JSON_TEST_PATH + PATH, 1e-##ACCURACY);\
}

#define CREATE_TESTS(NAME, PATH) \
    _CREATE_TEST(NAME, PATH, SequentialAlgorithm, 7);\
    _CREATE_TEST(NAME, PATH, OpenMPAlgorithm, 7); \
    _CREATE_TEST(NAME, PATH, CudaAlgorithm2_id, 7);

CREATE_TESTS(zero_timestamp, "00-zero_timestep.json");

CREATE_TESTS(1car_1step, "05-1car_1step.json");

CREATE_TESTS(1car_10steps, "10-1car_10steps.json");

CREATE_TESTS(speed_limit, "13-speed_limit.json");

CREATE_TESTS(1car_uturn, "15-1car_uturn.json");

CREATE_TESTS(3cars_1lane, "20-3cars_1lane.json");

CREATE_TESTS(2two_cars_before_lane_change, "25-2two_cars_before_lane_change.json");

CREATE_TESTS(2cars_lane_change, "27-2cars_lane_change.json");

CREATE_TESTS(2cars_lane_change_to_same_lane_one_car_faster, "30-cars_change_to_same_lane_one_car_faster.json");

CREATE_TESTS(2cars_lane_change_to_same_lane, "31-cars_change_to_same_lane.json");

CREATE_TESTS(lane_change_after_junction_to_missing_lane, "35-lane_change_after_junction.json");

CREATE_TESTS(lane_change_after_junction_to_same_lane, "36-lane_change_after_junction_same_lane.json");

CREATE_TESTS(cars_correct_turning, "40-cars_correct_turning.json");

CREATE_TESTS(tiny_100_steps, "42-tiny_100timestep.json");

CREATE_TESTS(4x4, "own_tests/4x4.json");

CREATE_TESTS(16x16, "own_tests/16x16.json");

/*TEST(Foo, Acceleration) {
    auto leadingCar = Car_id(0, 5, 30, 2, 2, 2, 2, 0.2, 0, 0, 0);
    auto followingCar = Car_id(1, 5, 30, 2, 2, 2, 2, 0.2, 0, 0, 0);

    auto a = followingCar.getAcceleration(&leadingCar);
    ASSERT_EQ(a, 2);
}*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}