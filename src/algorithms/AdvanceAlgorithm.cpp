//
// Created by oke on 15.12.18.
//

#include "algorithms/SequentialAlgorithm.h"
#include "algorithms/AdvanceAlgorithm.h"
#include "algorithms/OpenMPAlgorithm.h"



void AdvanceAlgorithm::registrate(std::string const & name, create_f * fp)
{
    registry()[name] = fp;
}

std::shared_ptr<AdvanceAlgorithm> AdvanceAlgorithm::instantiate(std::string const & name, Scenario* scenario)
{
    auto it = registry().find(name);
    return it == registry().end() ? nullptr : (it->second)(scenario);
}

std::unordered_map<std::string, AdvanceAlgorithm::create_f *> & AdvanceAlgorithm::registry()
{
    static std::unordered_map<std::string, AdvanceAlgorithm::create_f *> impl;
    return impl;
}

// TODO: Why doesnt this work in SequentialAlgorithm.h etc.
REGISTER_ALGORITHM(OpenMPAlgorithm);
REGISTER_ALGORITHM(SequentialAlgorithm);