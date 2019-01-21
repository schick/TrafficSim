//
// Created by maxi on 1/19/19.
//

#include "optimization/BaseOptimizer.h"

std::unordered_map<std::string, BaseOptimizer::create_f*> &BaseOptimizer::registry() {
    static std::unordered_map<std::string, create_f*> impl;
    return impl;
}

void BaseOptimizer::register_algorithm(const std::string &name, create_f *creator) {
    registry().insert({name, creator});
}

std::shared_ptr<BaseOptimizer> BaseOptimizer::instantiate(const std::string &name, nlohmann::json &scenarioData, const std::string &algorithm) {
    auto it = registry().find(name);
    return it == registry().end() ? nullptr : it->second(scenarioData, algorithm);
}
