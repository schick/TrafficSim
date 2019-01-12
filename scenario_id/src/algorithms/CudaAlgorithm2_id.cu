//
// Created by oke on 16.12.18.
//

#include <driver_types.h>
#include "algorithms/CudaAlgorithm2_id.h"
#include "cuda/cuda_utils.h"


__global__ void kernel_get_changes(Car_id::AdvanceData *change, CudaScenario_id * scenario_data,
        size_t *right_lane_neighbors, size_t *own_lane_neighbors, size_t *left_lane_neighbors) {
    size_t car_idx = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    if (car_idx < scenario_data->getNumCars()) {
        Lane_id::NeighboringObjects rightNeighbors(right_lane_neighbors[car_idx], right_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        Lane_id::NeighboringObjects ownNeighbors(own_lane_neighbors[car_idx], own_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        Lane_id::NeighboringObjects leftNeighbors(left_lane_neighbors[car_idx], left_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        change[car_idx] = algorithm_wrapper.nextStep(*scenario_data->getCar(car_idx), leftNeighbors, ownNeighbors, rightNeighbors);
    }
}

__global__ void kernel_apply_changes(Car_id::AdvanceData *change, CudaScenario_id * scenario_data) {
    size_t car_idx = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    if (car_idx < scenario_data->getNumCars()) {
        algorithm_wrapper.advanceStep(*scenario_data->getCar(change[car_idx].car), change[car_idx]);
    }
}

__global__ void kernel_update_signals(CudaScenario_id * scenario_data) {
    size_t jnt_idx = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    if (jnt_idx < scenario_data->getNumJunctions()) {
        algorithm_wrapper.updateSignals(*scenario_data->getJunction(jnt_idx));
    }
}

__device__ inline size_t gpu_ceil(double x) {
    if ((size_t) x == x) {
        return (size_t ) x;
    } else {
        return (size_t ) x + 1;
    }
}

#define C_MIN(a, b) (a < b ? a : b)
#define C_MAX(a, b) (a < b ? b : a)

__global__ void search_preceeding(CudaScenario_id *scenario, size_t *neighbors, int lane_offset, int clockRate) {
    size_t __beginn = clock();
    size_t debug_car = 7;
    extern __shared__ size_t temp_neighbors[];



    size_t section_id = threadIdx.x + blockDim.x * threadIdx.y;
    size_t car_idx = blockIdx.x + gridDim.y * blockIdx.y;
    size_t section_size = (size_t ) gpu_ceil((double)scenario->getNumCars() / (blockDim.y * blockDim.x));
    size_t section_count = gpu_ceil((double) scenario->getNumCars() / section_size);
    // if(car_idx == 1 && section_id == 0) printf("SectionId: %lu, SectionCount: %lu, SectionSize: %lu, Cars: %lu, blockDim: (%u, %u), gridDim: (%u, %u)\n",
    //          section_id, section_count, section_size, scenario->getNumCars(), blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    if (car_idx >= scenario->getNumCars())
        return;

    TrafficObject_id::Cmp cmp;

    const TrafficObject_id *closest_gt = nullptr;
    const TrafficObject_id *closest_lt = nullptr;

    Car_id *find_car = scenario->getCar(car_idx);

    Lane_id *lane = scenario->getLane(find_car->lane);
    AlgorithmWrapper algo(*scenario);
    int lane_offset_cnt = lane_offset;

    size_t start = clock();
    while(lane_offset_cnt != 0 && lane != nullptr) {
        Road_id::NeighboringLanes neigh_lanes = algo.getNeighboringLanes(*lane);
        if (lane_offset_cnt > 0) {
            if(neigh_lanes.right != (size_t) -1)
                lane = scenario->getLane(neigh_lanes.right);
            else
                lane = nullptr;
            lane_offset_cnt--;
        } else {
            if(neigh_lanes.left != (size_t) -1)
                lane = scenario->getLane(neigh_lanes.left);
            else
                lane = nullptr;
            lane_offset_cnt++;
        }
    }

    size_t after_first_while = clock();
    if (lane == nullptr) {
        neighbors[car_idx] = (size_t) -1;
        neighbors[car_idx + scenario->getNumCars()] = (size_t) -1;
        return;
    }
    size_t lane_id = lane->id;

    for(size_t cmp_idx=section_size * section_id; cmp_idx < C_MIN(section_size * (section_id + 1), scenario->getNumCars()); cmp_idx++) {
        assert(cmp_idx < scenario->getNumCars());
        Car_id *car = scenario->getCar(cmp_idx);
        if(car->lane == lane_id && car != find_car) {
            if(cmp(car, find_car)) {
                // objects behind
                if(closest_lt == nullptr || !cmp(car, closest_lt)) {
                    closest_lt = car;
                }
            } else {

                // objects in front of
                if(closest_gt == nullptr || cmp(car, closest_gt)) {
                    closest_gt = car;
                }
            }
        }
    }
    size_t after_second_for = clock();

    if(closest_gt != nullptr) {
        temp_neighbors[section_id] = closest_gt->id;
        //result_ids[section_id] = closest_gt->id;
    } else {
        temp_neighbors[section_id] = (size_t) -1;
    }

    if(closest_lt != nullptr) {
        temp_neighbors[section_id + section_count] = closest_lt->id;
        // result_ids[section_id * 2] = closest_lt->id;
    } else {
        temp_neighbors[section_id + section_count] = (size_t) -1;
    }

    __syncthreads();

    constexpr size_t each = 2;
    size_t reduce_id = threadIdx.x + blockDim.x * threadIdx.y;
    size_t reduce_size = each;
    size_t reduce_count = gpu_ceil((double) section_count / reduce_size);


    while(reduce_size / each <= section_count) {
        Car_id *car = nullptr;
        Car_id *car2 = nullptr;
        for(size_t r_idx = reduce_id * reduce_size; r_idx < C_MIN((reduce_id + 1) * reduce_size, section_count); r_idx += reduce_size / each) {

            if (temp_neighbors[r_idx] != (size_t) -1) {
                Car_id *cmp_car = scenario->getCar(temp_neighbors[r_idx]);
                if(car == nullptr || cmp(cmp_car, car)) {
                    car = cmp_car;
                }
            }

            if (temp_neighbors[r_idx + section_count] != (size_t) -1) {
                Car_id *cmp_car = scenario->getCar(temp_neighbors[r_idx + section_count]);
                if(car2 == nullptr || !cmp(cmp_car, car2)) {
                    car2 = cmp_car;
                }
            }
        }

        if(car != nullptr)
            temp_neighbors[reduce_id * reduce_size] = car->id;
            if(car2 != nullptr)
            temp_neighbors[reduce_id * reduce_size + section_count] = car2->id;
        reduce_size *= each;

        __syncthreads();
    }
    __syncthreads();

    size_t after_third_while = clock();
    if (section_id == 0) {
        Lane_id &l = *scenario->getLane(find_car->lane);
        Road_id &r = *scenario->getRoad(l.road);
        Junction_id &j = *scenario->getJunction(r.to);
        RedTrafficLight_id *light = scenario->getLight(j.red_traffic_lights_ids[(r.roadDir + 2) % 4][l.lane_num + lane_offset]);
        if(light->lane != -1) {
            assert(light->lane == lane_id);
            if((temp_neighbors[0] == (size_t) -1 || cmp(light, scenario->getCar(temp_neighbors[0]))) && cmp(find_car, light)) {
                temp_neighbors[0] = j.red_traffic_lights_ids[(r.roadDir + 2) % 4][l.lane_num + lane_offset];
            }
            if((temp_neighbors[section_count] == (size_t) -1 || cmp(scenario->getCar(temp_neighbors[section_count]), light)) && cmp(light, find_car)) {
                temp_neighbors[section_count] = j.red_traffic_lights_ids[(r.roadDir + 2) % 4][l.lane_num + lane_offset];
            }
        }
        neighbors[car_idx] = temp_neighbors[0];
        neighbors[car_idx + scenario->getNumCars()] = temp_neighbors[section_count];
    }
    size_t __end = clock();
    if(car_idx == 0 && section_id == 0) printf("Total: %fms, \n    1: %fms, \n    2: %fms, \n    3: %fms\n\n", (float) (__end - __beginn) / clockRate,
                            (float) (after_first_while - start) / clockRate,
                            (float) (after_second_for - after_first_while) / clockRate,
                            (float) (after_third_while - after_second_for) / clockRate);
}


void test_left_lane_neighbors(CudaScenario_id * device_scenario, Scenario_id *s) {

    CudaScenario_id scenario = CudaScenario_id::fromScenarioData(*s);
    AlgorithmWrapper algorithmWrapper(scenario);
    dim3 blocks(ceil(sqrt(scenario.getNumCars())), ceil(sqrt(scenario.getNumCars())));    /* Number of blocks   */
    dim3 threads(1, 10);  /* Number of threads  */

    size_t *dev_neighbors;

    gpuErrchk(cudaMalloc((void**) &dev_neighbors, 2 * scenario.getNumCars() * sizeof(size_t)));


    search_preceeding<<<blocks, threads, 2 * threads.x * threads.y * sizeof(size_t)>>>(device_scenario, dev_neighbors, -1, -1);

    gpuErrchk( cudaPeekAtLastError() );

    cudaDeviceSynchronize();
    size_t neighbors[2 * scenario.getNumCars()];
    gpuErrchk(cudaMemcpy(neighbors, dev_neighbors,  2 * scenario.getNumCars() * sizeof(size_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < scenario.getNumCars(); i++) {
        Road_id::NeighboringLanes lanes = algorithmWrapper.getNeighboringLanes(*scenario.getLane(scenario.getCar(i)->lane));
        if(lanes.left == (size_t) -1) {
            assert(neighbors[scenario.getNumCars() + i] == (size_t) -1);
            assert(neighbors[i] == (size_t) -1);
            continue;
        }
        Lane_id::NeighboringObjects neig = algorithmWrapper.getNeighboringObjects(*scenario.getCar(i), *scenario.getLane(lanes.left));


        if (neig.back == -1) {
            assert(neighbors[scenario.getNumCars() + i] == (size_t) -1);
        } else {
            assert(neighbors[scenario.getNumCars() + i] != (size_t) -1 && neighbors[scenario.getNumCars() + i] == neig.back);
        }
        if (neig.front == -1) {
            assert(neighbors[i] == (size_t) -1);
        } else {
            assert(neighbors[i] != (size_t) -1 && neighbors[i] == neig.front);
        }
    }

    printf("passed left_lane test.\n");
}



void CudaAlgorithm2_id::advance(size_t steps) {
    //test_own_lane_neighbors(device_cuda_scenario, getIDScenario());
    //test_left_lane_neighbors(device_cuda_scenario, getIDScenario());
    //test_right_lane_neighbors(device_cuda_scenario, getIDScenario());

    CudaScenario_id scenario = CudaScenario_id::fromScenarioData(*getIDScenario());

    printf("Cars: %lu\n", scenario.getNumCars());
    size_t *left_lane_neighbors, *right_lane_neighbors, *own_lane_neighbors;

    gpuErrchk(cudaMalloc((void**) &left_lane_neighbors, 2 * scenario.getNumCars() * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void**) &right_lane_neighbors, 2 * scenario.getNumCars() * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void**) &own_lane_neighbors, 2 * scenario.getNumCars() * sizeof(size_t)));


    dim3 blocks(ceil(sqrt(scenario.getNumCars())), ceil(sqrt(scenario.getNumCars())));    /* Number of blocks   */
    dim3 threads(ceil(sqrt(scenario.getNumCars())), ceil(sqrt(scenario.getNumCars())));  /* Number of threads  */

    threads.x = threads.y = 25;
    printf("SHARED: %f\n", 2 * threads.x * threads.y * sizeof(size_t) / 1024.);

    Car_id::AdvanceData *device_changes;
    gpuErrchk(cudaMalloc((void**) &device_changes, getIDScenario()->cars.size() * sizeof(Car_id::AdvanceData)));

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clockRate = prop.clockRate;
    cudaError_t err = cudaDeviceGetAttribute(&prop.clockRate, cudaDevAttrClockRate, 0);
    cudaEvent_t starts[steps], stops[steps];
    for(int i = 0; i < steps; i++) {
        cudaEventCreate(&starts[i]);
        cudaEventCreate(&stops[i]);
    }

    for(int i = 0; i < steps; i++) {
        search_preceeding<<<blocks, threads, 2 * threads.x * threads.y * sizeof(size_t)>>>(device_cuda_scenario, right_lane_neighbors, 1, clockRate);
        gpuErrchk( cudaPeekAtLastError() );
        cudaEventRecord(starts[i]);
        search_preceeding<<<blocks, threads, 2 * threads.x * threads.y * sizeof(size_t)>>>(device_cuda_scenario, own_lane_neighbors, 0, clockRate);
        cudaEventRecord(stops[i]);
        gpuErrchk( cudaPeekAtLastError() );
        search_preceeding<<<blocks, threads, 2 * threads.x * threads.y * sizeof(size_t)>>>(device_cuda_scenario, left_lane_neighbors, -1, clockRate);
        gpuErrchk( cudaPeekAtLastError() );

        kernel_get_changes<<<512, 512>>>(device_changes, device_cuda_scenario, right_lane_neighbors, own_lane_neighbors, left_lane_neighbors);
        gpuErrchk( cudaPeekAtLastError() );
        kernel_apply_changes<<<512, 512>>>(device_changes, device_cuda_scenario);
        gpuErrchk( cudaPeekAtLastError() );
        kernel_update_signals<<<512, 512>>>(device_cuda_scenario);
        gpuErrchk( cudaPeekAtLastError() );
    }
    printf("clock: %d\n", clockRate);

    cudaDeviceSynchronize();
    cudaEventSynchronize(stops[steps - 1]);
    double total_ms = 0.;
    for(int i = 0; i < steps; i++) {
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, starts[i], stops[i]);
        total_ms += (double) milliseconds / steps;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, starts[steps - 1], stops[steps - 1]);
    printf("last duration: %f\n", milliseconds);
    cudaEventElapsedTime(&milliseconds, starts[steps - 2], stops[steps - 2]);
    printf("-2-last duration: %f\n", milliseconds);
    printf("average duration: %f\n", total_ms);
    gpuErrchk( cudaPeekAtLastError() );

    device_cuda_scenario->retriveData(getIDScenario());

    gpuErrchk(cudaFree(device_changes));
}