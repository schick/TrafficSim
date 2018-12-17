//
// Created by oke on 16.12.18.
//

#include "algorithms/cuda_algorithms.cuh"

#include <iostream>
#include "../../../scenario_ref/include/Scenario.h"
#include <math.h>

// function to add the elements of two arrays

__global__
void do_something(int n, Car *cars) {

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    cars[i].x += 1;
}

void next(Scenario &s) {/**
    int max = 0;
    for(auto &c : s.cars) max = std::max(c.first, max);
    Car *cudaCars;
    Car *cars = (Car*)malloc(max * sizeof(Car));
    printf("max: %d\n", max);
    for(int i=0; i < max; i++) {
        printf("%d\n", i);
        printf("%p\n", cars);
        if (s.cars.find(i) != s.cars.end()) {
            printf("%.2f\n", s.cars[i].x);
            cars[i] = Car();
        }
    }
    cudaMalloc(&cudaCars, max * sizeof(Car));
    cudaMemcpy(cudaCars, cars, max*sizeof(Car), cudaMemcpyHostToDevice);
    for(int i =0; i<10; i++)
        printf("%.2f\n", cars[i].x);
    do_something<<<(max+255)/256, 256>>>(max, cudaCars);

    cudaMemcpy(cars, cudaCars, max*sizeof(Car), cudaMemcpyDeviceToHost);
    for(int i =0; i<10; i++)
        printf("%d: %.2f\n", i, cars[i].x);
    printf("cuda done.\n");*/
}
