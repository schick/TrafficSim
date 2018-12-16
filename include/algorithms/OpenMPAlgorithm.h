//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_OPENMPALGORITHM_H
#define PROJECT_OPENMPALGORITHM_H

#include "AdvanceAlgorithm.h"

class OpenMPAlgorithm : public AdvanceAlgorithm {

public:

    explicit OpenMPAlgorithm(Scenario *scenario) : AdvanceAlgorithm(scenario) {};

    static std::shared_ptr<AdvanceAlgorithm> create(Scenario* scenario) { return std::make_shared<OpenMPAlgorithm>(scenario); }

    void advance(size_t steps) override;
};


#endif //PROJECT_OPENMPALGORITHM_H
