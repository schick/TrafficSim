//
// Created by oke on 29.01.19.
//

#ifndef TRAFFIC_SIM_REFALGORITHM_H
#define TRAFFIC_SIM_REFALGORITHM_H


#include "AdvanceAlgorithm.h"
#include "BaseScenario.h"
#include "model/Scenario.h"

class RefAlgorithm : public AdvanceAlgorithm {
private:
    Scenario *ref_scenario;
public:
    explicit RefAlgorithm(std::shared_ptr<BaseScenario> scenario)
        : AdvanceAlgorithm(scenario), ref_scenario(dynamic_cast<Scenario*>(getScenario().get())) {};

    Scenario* getRefScenario() {
        return ref_scenario;
    }

};

#endif //TRAFFIC_SIM_REFALGORITHM_H
