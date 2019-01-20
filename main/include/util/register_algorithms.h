//
// Created by oke on 17.12.18.
//

#ifndef TRAFFIC_SIM_ALGORITHMS_H
#define TRAFFIC_SIM_ALGORITHMS_H

#include "algorithms/SequentialAlgorithm.h"
#include "algorithms/OpenMPAlgorithm.h"
#include "algorithms/TestAlgo.h"

#include "optimization/BaseOptimizer.h"
#include "optimization/RandomOptimizer.h"

REGISTER_ALGORITHM(OpenMPAlgorithm);
REGISTER_ALGORITHM(SequentialAlgorithm);

REGISTER_OPTIMIZER(RandomOptimizer);

#ifdef WITH_CUDA
    REGISTER_ALGORITHM(TestAlgo);
    //REGISTER_ALGORITHM(CudaAlgorithm2_id);
#endif

#ifdef ALL_ALGORITHMS
    REGISTER_ALGORITHM(SequentialAlgorithm_id);
    REGISTER_ALGORITHM(OpenMPAlgorithm_id);
#endif

#endif //TRAFFIC_SIM_ALGORITHMS_H
