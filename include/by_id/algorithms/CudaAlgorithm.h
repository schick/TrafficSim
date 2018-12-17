//
// Created by oke on 16.12.18.
//

#ifndef TRAFFIC_SIM_CUDAALGORITHM_H
#define TRAFFIC_SIM_CUDAALGORITHM_H

#include "algorithms/AdvanceAlgorithm.h"

class CudaAlgorithm : public AdvanceAlgorithm {
public:
    explicit CudaAlgorithm(Scenario *scenario) : AdvanceAlgorithm(scenario) {};

    static std::shared_ptr<AdvanceAlgorithm> create(Scenario* scenario) { return std::make_shared<CudaAlgorithm>(scenario); }

    void advance(size_t steps) override;


};

#endif //TRAFFIC_SIM_CUDAALGORITHM_H
