//
// Created by oke on 17.12.18.
//

#ifndef TRAFFIC_SIM_ALGORITHMS_H
#define TRAFFIC_SIM_ALGORITHMS_H

#include "algorithms/SequentialAlgorithm.h"
#include "algorithms/OpenMPAlgorithm.h"
#include "algorithms/CudaAlgorithm.h"

#include "optimization/ParallelRandomOptimizer.h"
#include "optimization/SequentialRandomOptimizer.h"
#include "optimization/GeneticOptimizer.h"
#include "optimization/DistributionOptimizer.h"

REGISTER_ALGORITHM(OpenMPAlgorithm);
REGISTER_ALGORITHM(SequentialAlgorithm);

REGISTER_OPTIMIZER(ParallelRandomOptimizer);
REGISTER_OPTIMIZER(SequentialRandomOptimizer);
REGISTER_OPTIMIZER(GeneticOptimizer);
REGISTER_OPTIMIZER(DistributionOptimizer);

#ifdef WITH_CUDA
    REGISTER_ALGORITHM(CudaAlgorithm);
#endif

#ifdef ALL_ALGORITHMS
    REGISTER_ALGORITHM(SequentialAlgorithm_id);
    REGISTER_ALGORITHM(OpenMPAlgorithm_id);
#endif

#endif //TRAFFIC_SIM_ALGORITHMS_H
