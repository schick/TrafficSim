//
// Created by oke on 15.12.18.
//

#include "AdvanceAlgorithm.h"
#include "util/json.hpp"

void AdvanceAlgorithm::registrate(std::string const & name, create_f * fp, create_scenario_f * sc, create_optimization_scenario_f *osc)
{
    registry()[name] = {fp, sc, osc};
}

std::shared_ptr<AdvanceAlgorithm> AdvanceAlgorithm::instantiate(std::string const & name, json &scenario_data) {
    auto it = registry().find(name);
    std::shared_ptr<BaseScenario> scenario = it == registry().end() ? nullptr : (it->second).scenario_creator();
    if (scenario.get() == nullptr) return nullptr;
    scenario->parse(scenario_data);
    return it == registry().end() ? nullptr : (it->second).creater(scenario);
}

std::shared_ptr<AdvanceAlgorithm> AdvanceAlgorithm::instantiateOptimization(std::string const &name, json &scenario_data) {
    auto it = registry().find(name);
    std::shared_ptr<BaseScenario> scenario = it == registry().end() ? nullptr : (it->second).optimization_scenario_creator();
    if (scenario.get() == nullptr) return nullptr;
    scenario->parse(scenario_data);
    return it == registry().end() ? nullptr : (it->second).creater(scenario);
}

std::unordered_map<std::string, AdvanceAlgorithm::creator_struct> & AdvanceAlgorithm::registry()
{
    static std::unordered_map<std::string, AdvanceAlgorithm::creator_struct> impl;
    return impl;
}
