//
// Created by oke on 17.12.18.
//

#ifndef TRAFFIC_SIM_ALGORITHMS_H
#define TRAFFIC_SIM_ALGORITHMS_H

#include "../scenario_id/include/algorithms/SequentialAlgorithm_id.h"
#include "../scenario_id/include/algorithms/OpenMPAlgorithm_id.h"
#include "../scenario_ref/include/algorithms/SequentialAlgorithm.h"
#include "../scenario_ref/include/algorithms/OpenMPAlgorithm.h"

#include "../scenario_id/include/algorithms/CudaAlgorithm_id.h"

REGISTER_ALGORITHM(SequentialAlgorithm);
REGISTER_ALGORITHM(OpenMPAlgorithm);

#ifdef WITH_CUDA
    REGISTER_ALGORITHM(CudaAlgorithm_id);
#endif

#ifdef ALL_ALGORITHMS
    REGISTER_ALGORITHM(SequentialAlgorithm_id);
    REGISTER_ALGORITHM(OpenMPAlgorithm_id);
#endif

#endif //TRAFFIC_SIM_ALGORITHMS_H
