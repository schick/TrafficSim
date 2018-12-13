
#include <fstream>
#include <gtest/gtest.h>

#include "json.hpp"
#include "Scenario.h"


using json = nlohmann::json;

void test_file(std::string fn, double genauigkeit) {

    json input, loesung;

    // read input file
    std::ifstream json_file(fn);
    EXPECT_NO_THROW(json_file >> input);
    // read loesung
    std::ifstream json_file_out(fn + ".sol");
    EXPECT_NO_THROW(json_file_out >> loesung);

    Scenario scenario;
    scenario.parse(input);

    OkesExampleAdvanceAlgorithm advancer(&scenario);

    for (int i = 0; i < input["time_steps"]; i++) {
        advancer.advance();
    }

    json output = scenario.toJson();

    //ASSERT_TRUE(loesung == output);
    ASSERT_EQ(loesung["cars"].size(), output["cars"].size());
    for (auto &car_json : output["cars"]) {
        bool found_car = false;
        for (auto cmp_car_json : loesung["cars"]) {
            if (cmp_car_json["id"] == car_json["id"] &&
                cmp_car_json["to"] == car_json["to"] &&
                cmp_car_json["from"] == car_json["from"] &&
                abs((double)cmp_car_json["position"] - (double)car_json["position"]) <= genauigkeit &&
                cmp_car_json["lane"] == car_json["lane"]) {
                found_car = true;
                break;
            }
        }
        ASSERT_TRUE(found_car);
    }
}

#define JSON_TEST_PATH std::string("tests/")

TEST(TestJson, zero_timestamp) {
    test_file(JSON_TEST_PATH + "00-zero_timestep.json", 1e-7);
}

TEST(TestJson, 1car_1step) {
    test_file(JSON_TEST_PATH + "05-1car_1step.json", 1e-7);
}

TEST(TestJson, 1car_10steps) {
    test_file(JSON_TEST_PATH + "10-1car_10steps.json", 1e-7);
}

TEST(TestJson, speed_limit) {
    test_file(JSON_TEST_PATH + "13-speed_limit.json", 1e-7);
}

TEST(TestJson, 1car_uturn) {
    test_file(JSON_TEST_PATH + "15-1car_uturn.json", 1e-7);
}

TEST(TestJson, 3cars_1lane) {
    test_file(JSON_TEST_PATH + "20-3cars_1lane.json", 1e-7);
}

TEST(TestJson, 2two_cars_before_lane_change) {
    test_file(JSON_TEST_PATH + "25-2two_cars_before_lane_change.json", 1e-7);
}

TEST(TestJson, 2cars_lane_change) {
    test_file(JSON_TEST_PATH + "27-2cars_lane_change.json", 1e-7);
}

TEST(TestJson, 2cars_lane_change_to_same_lane) {
    test_file(JSON_TEST_PATH + "30-cars_change_to_same_lane_one_car_faster.json", 1e-7);
}

TEST(TestJson, lane_change_after_junction_to_missing_lane) {
    test_file(JSON_TEST_PATH + "35-lane_change_after_junction.json", 1e-7);
}

TEST(TestJson, lane_change_after_junction_to_same_lane) {
    test_file(JSON_TEST_PATH + "36-lane_change_after_junction_same_lane.json", 1e-7);
}

TEST(TestJson, cars_correct_turning) {
    test_file(JSON_TEST_PATH + "40-cars_correct_turning.json", 1e-7);
}

TEST(TestJson, tiny_400_steps) {
    test_file(JSON_TEST_PATH + "42-tiny_100timestep.json", 1e-7);
}



TEST(TestJsonExact, zero_timestamp) {
    test_file(JSON_TEST_PATH + "00-zero_timestep.json", 0);
}

TEST(TestJsonExact, 1car_1step) {
    test_file(JSON_TEST_PATH + "05-1car_1step.json", 0);
}

TEST(TestJsonExact, 1car_10steps) {
    test_file(JSON_TEST_PATH + "10-1car_10steps.json", 0);
}

TEST(TestJsonExact, speed_limit) {
    test_file(JSON_TEST_PATH + "13-speed_limit.json", 0);
}

TEST(TestJsonExact, 1car_uturn) {
    test_file(JSON_TEST_PATH + "15-1car_uturn.json", 0);
}

TEST(TestJsonExact, 3cars_1lane) {
    test_file(JSON_TEST_PATH + "20-3cars_1lane.json", 0);
}

TEST(TestJsonExact, 2two_cars_before_lane_change) {
    test_file(JSON_TEST_PATH + "25-2two_cars_before_lane_change.json", 0);
}

TEST(TestJsonExact, 2cars_lane_change) {
    test_file(JSON_TEST_PATH + "27-2cars_lane_change.json", 0);
}

TEST(TestJsonExact, 2cars_lane_change_to_same_lane) {
    test_file(JSON_TEST_PATH + "30-cars_change_to_same_lane_one_car_faster.json", 0);
}

TEST(TestJsonExact, lane_change_after_junction_to_missing_lane) {
    test_file(JSON_TEST_PATH + "35-lane_change_after_junction.json", 0);
}

TEST(TestJsonExact, lane_change_after_junction_to_same_lane) {
    test_file(JSON_TEST_PATH + "36-lane_change_after_junction_same_lane.json", 0);
}

TEST(TestJsonExact, cars_correct_turning) {
    test_file(JSON_TEST_PATH + "40-cars_correct_turning.json", 0);
}

TEST(TestJsonExact, tiny_400_steps) {
    test_file(JSON_TEST_PATH + "42-tiny_100timestep.json", 0);
}
/*TEST(Foo, Acceleration) {
    auto leadingCar = Car(0, 5, 30, 2, 2, 2, 2, 0.2, 0, 0, 0);
    auto followingCar = Car(1, 5, 30, 2, 2, 2, 2, 0.2, 0, 0, 0);

    auto a = followingCar.getAcceleration(&leadingCar);
    ASSERT_EQ(a, 2);
}*/

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}