//
// Created by maxi on 1/19/19.
//

#ifndef TRAFFIC_SIM_BASEOPTIMIZER_H
#define TRAFFIC_SIM_BASEOPTIMIZER_H


#include "util/json.hpp"
#include "AdvanceAlgorithm.h"

class BaseOptimizer {

public:
    BaseOptimizer(nlohmann::json &scenarioData, const std::string &algorithm): scenarioData(scenarioData), algorithm(algorithm) {
        minTravelLength = scenarioData["min_travel_distance"];
    }

    bool hasValidAlgorithm() {
        std::vector<std::string> algorithms = AdvanceAlgorithm::getAlgorithms();
        auto it = std::find(algorithms.begin(), algorithms.end(), algorithm);
        return it != algorithms.end();
    }

    virtual nlohmann::json optimize() = 0;

    /** simple registry */
    using create_f = std::shared_ptr<BaseOptimizer>(nlohmann::json &scenarioData, const std::string &algorithm);
    static void register_algorithm(const std::string &name, create_f *creator);
    static std::shared_ptr<BaseOptimizer> instantiate(const std::string &name, nlohmann::json &scenarioData, const std::string &algorithm);

protected:

    nlohmann::json &scenarioData;
    std::string algorithm;

    double minTravelLength;

private:
    static std::unordered_map<std::string, create_f*> &registry();

};


template <typename D>
class DefaultConstructorRegistrar {
public:
    static std::shared_ptr<BaseOptimizer> default_creator(nlohmann::json &scenarioData, const std::string &algorithm) {
        return std::make_shared<D>(scenarioData, algorithm);
    }

    explicit DefaultConstructorRegistrar(const std::string &name) {
        BaseOptimizer::register_algorithm(name, &DefaultConstructorRegistrar<D>::default_creator);
    }
};

#define STR(str) #str
#define REGISTER_OPTIMIZER(CLASS) DefaultConstructorRegistrar<CLASS> CLASS##_registrar(STR(CLASS))


#endif //TRAFFIC_SIM_BASEOPTIMIZER_H
