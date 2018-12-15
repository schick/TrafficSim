//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_SCENARIO_H
#define PROJECT_SCENARIO_H

#include <memory>
#include <omp.h>

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

    explicit AdvanceAlgorithm(Scenario *scenario) : scenario(scenario) {};

    virtual std::vector<Car::AdvanceData> calculateCarChanges() = 0;

    virtual void advance(size_t steps = 1);

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
    void initJunctions();
    void parseCars(json & input);
    void parseRoads(json & input);
    void createRoads(const nlohmann::json & road);
    void createLanesForRoad(const nlohmann::json & road, std::unique_ptr<Road> &road_obj);
    void parseJunctions(json &input);
    json toJson();

};


class SequentialAlgorithm : public AdvanceAlgorithm {

public:

    explicit SequentialAlgorithm(Scenario *scenario) : AdvanceAlgorithm(scenario) {};

    std::vector<Car::AdvanceData> calculateCarChanges() override {

        std::vector<Car::AdvanceData> changes;
        for (std::unique_ptr<Car> &c : getScenario()->cars) {
            changes.emplace_back(c->nextStep());
        }
        return changes;
    };

    void advanceCars() override {
        std::vector<Car::AdvanceData> changes = calculateCarChanges();
        for (Car::AdvanceData &d : changes) {
            d.car->advanceStep(d);
        }
    }

    void advanceTrafficLights() override {
        for (std::unique_ptr<Junction> &j : getScenario()->junctions) {
            j->updateSignals();
        }
    }

};


class OpenMPAlgorithm : public AdvanceAlgorithm {

public:

    explicit OpenMPAlgorithm(Scenario *scenario) : AdvanceAlgorithm(scenario) {};

    std::vector<Car::AdvanceData> calculateCarChanges() override {
        std::vector<Car::AdvanceData> changes(getScenario()->cars.size());
        auto &cars = getScenario()->cars;
        #pragma for shared(changes, cars)
        for (int i = 0; i < cars.size(); i++) {
            changes[i] = cars[i]->nextStep();
        }
        return changes;
    };

    void advanceCars() override {
    }


    void advance(size_t steps = 1) override {

        std::vector<Car::AdvanceData> changes = calculateCarChanges();
        auto &cars = getScenario()->cars;
        #pragma omp parallel shared(cars, changes)
        {
            for (int i = 0; i < steps; i++) {

                #pragma omp for
                for (int i = 0; i < cars.size(); i++) {
                    changes[i] = cars[i]->nextStep();
                }

                #pragma omp for
                for (int i = 0; i < changes.size(); i++) {
                    changes[i].car->advanceStep(changes[i]);
                }
            }
        }
    }

    void advanceTrafficLights() override {
        for (std::unique_ptr<Junction> &j : getScenario()->junctions) {
            j->updateSignals();
        }
    }

};



#endif //PROJECT_SCENARIO_H
