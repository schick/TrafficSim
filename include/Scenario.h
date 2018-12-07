//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H

#include <memory>

#include "json.hpp"

#include "Car.h"
#include "Junction.h"
#include "Lane.h"
#include "RedTrafficLight.h"
#include "Road.h"


using json = nlohmann::json;


class Scenario;

class AdvanceAlgorithm {
private:

    Scenario *scenario;

public:

    Scenario *getScenario() { return scenario; }

    AdvanceAlgorithm(Scenario *scenario) : scenario(scenario) {};

    virtual std::vector<Car::AdvanceData> calculateCarChanges() = 0;

    void advance(size_t steps=1);

    virtual void advanceCars() = 0;
    virtual void advanceTrafficLights() = 0;
};


class Scenario {
public:

    std::vector<std::unique_ptr<Junction>> junctions;
    std::vector<std::unique_ptr<Road>> roads;
    std::vector<std::unique_ptr<Lane>> lanes;
    std::vector<std::unique_ptr<Car>> cars;

    void parse(json input);
    json toJson();

};


class OkesExampleAdvanceAlgorithm : public AdvanceAlgorithm {

public:

    explicit OkesExampleAdvanceAlgorithm(Scenario *scenario) : AdvanceAlgorithm(scenario) {};

    std::vector<Car::AdvanceData> calculateCarChanges() override {
        std::vector<Car::AdvanceData> changes;
        for (std::unique_ptr<Car> &c : getScenario()->cars) {
            changes.emplace_back(c->nextStep());
        }
        return changes;
    };

    void advanceCars() override {
        std::vector<Car::AdvanceData> changes = calculateCarChanges();
        for(Car::AdvanceData &d : changes) {
            d.car->advanceStep(d);
        }
    }

    void advanceTrafficLights() override {

    }

};


#endif //PROJECT_SCENARIO_H
