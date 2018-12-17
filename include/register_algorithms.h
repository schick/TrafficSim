//
// Created by oke on 17.12.18.
//

#ifndef TRAFFIC_SIM_ALGORITHMS_H
#define TRAFFIC_SIM_ALGORITHMS_H

#include "algorithms/SequentialAlgorithm.h"
#include "algorithms/SequentialAlgorithm_id.h"
#include "algorithms/OpenMPAlgorithm.h"
#include "algorithms/OpenMPAlgorithm_id.h"

REGISTER_ALGORITHM(SequentialAlgorithm);
REGISTER_ALGORITHM(OpenMPAlgorithm);

#ifdef ALL_ALGORITHMS
    REGISTER_ALGORITHM(SequentialAlgorithm_id);
    REGISTER_ALGORITHM(OpenMPAlgorithm_id);
#endif

#endif //TRAFFIC_SIM_ALGORITHMS_H
