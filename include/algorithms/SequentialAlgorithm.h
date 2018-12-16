//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_SEQUENTIALALGORITHM_H
#define PROJECT_SEQUENTIALALGORITHM_H

#include "AdvanceAlgorithm.h"


class SequentialAlgorithm : public AdvanceAlgorithm {

public:

    static std::shared_ptr<AdvanceAlgorithm> create(Scenario* scenario) { return std::make_shared<SequentialAlgorithm>(scenario); }

    explicit SequentialAlgorithm(Scenario *scenario) : AdvanceAlgorithm(scenario) {};

    std::vector<Car::AdvanceData> calculateCarChanges();

    void advanceCars();
    void advanceTrafficLights();
    void advance(size_t steps) override;

};




#endif //PROJECT_SEQUENTIALALGORITHM_H
