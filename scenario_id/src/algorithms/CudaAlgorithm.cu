//
// Created by oke on 09.01.19.
//

#include <curand_mtgp32_kernel.h>
#include <model/Lane_id.h>
#include "algorithms/CudaAlgorithm.h"
#include "cuda_utils/cuda_utils.h"
#include <chrono>
#include <thread>

#include "cuda_utils/SortedBucketContainer.h"
#include "cuda_utils/PreScan.h"
#include "cuda_utils/SortBuffer.h"

__device__ void test_right_lane_neighbors(TrafficObject_id **neighbors, CudaScenario_id *scenario) {
    AlgorithmWrapper algorithmWrapper(*scenario);
    size_t car_id = GetGlobalIdx();
    if(car_id == 0) printf("test_right_lane_neighbors\n");
    if (car_id >= scenario->getNumCars()) return;


    Road_id::NeighboringLanes lanes = algorithmWrapper.getNeighboringLanes(*scenario->getLane(scenario->getCar(car_id)->lane));
    if(lanes.right == (size_t) -1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
        assert(neighbors[car_id] == nullptr);
        return;
    }
    Lane_id::NeighboringObjects neig = algorithmWrapper.getNeighboringObjects(*scenario->getCar(car_id), *scenario->getLane(lanes.right));

    if (neig.back == (size_t )-1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
    } else {
        assert(neighbors[scenario->getNumCars() +car_id ] != nullptr && neighbors[scenario->getNumCars() +car_id]->id == neig.back);
    }

    if (neig.front ==  (size_t )-1) {
        if(neighbors[car_id] != nullptr) printf("%lu\n", neighbors[car_id]->id);

        assert(neighbors[car_id] == nullptr);
    } else {

        if(!(neighbors[car_id] != nullptr && neighbors[car_id]->id == neig.front)) {
            printf("Car(%lu, %lu): %lu == %lu, tllane(%lu)\n", car_id, scenario->getCar(car_id)->lane, neighbors[car_id] != nullptr ? neighbors[car_id]->id : (size_t ) -1, neig.front, scenario->getLight(neig.front)->lane);
        }
        assert(neighbors[car_id] != nullptr && neighbors[car_id]->id == neig.front);
    }
}

__device__ void test_left_lane_neighbors(TrafficObject_id **neighbors, CudaScenario_id *scenario) {
    AlgorithmWrapper algorithmWrapper(*scenario);
    size_t car_id = GetGlobalIdx();
    if(car_id == 0) printf("test_left_lane_neighbors\n");
    if (car_id >= scenario->getNumCars()) return;


    Road_id::NeighboringLanes lanes = algorithmWrapper.getNeighboringLanes(*scenario->getLane(scenario->getCar(car_id)->lane));
    if(lanes.left == (size_t) -1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
        assert(neighbors[car_id] == nullptr);
        return;
    }
    Lane_id::NeighboringObjects neig = algorithmWrapper.getNeighboringObjects(*scenario->getCar(car_id), *scenario->getLane(lanes.left));

    if (neig.back == (size_t )-1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
    } else {
        assert(neighbors[scenario->getNumCars() +car_id ] != nullptr && neighbors[scenario->getNumCars() +car_id]->id == neig.back);
    }

    if (neig.front ==(size_t ) -1) {
        assert(neighbors[car_id] == nullptr);
    } else {
        assert(neighbors[car_id] != nullptr && neighbors[car_id]->id == neig.front);
    }
}

__device__ void test_own_lane_neighbors(TrafficObject_id **neighbors, CudaScenario_id *scenario) {
    AlgorithmWrapper algorithmWrapper(*scenario);
    size_t car_id = GetGlobalIdx();
    if(car_id == 0) printf("test_own_lane_neighbors\n");

    if (car_id >= scenario->getNumCars()) return;

    Lane_id::NeighboringObjects neig = algorithmWrapper.getNeighboringObjects(*scenario->getCar(car_id), *scenario->getLane(scenario->getCar(car_id)->lane));

    if (neig.back == (size_t )-1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
    } else {
        if(!(neighbors[scenario->getNumCars() +car_id] != nullptr && neighbors[scenario->getNumCars() +car_id]->id == neig.back)) {
            printf("c - %5lu, FrontCar(%5lu/%5lu) at Lane(%lu)\n", car_id, neig.back, scenario->getNumCars(), scenario->getCar(neig.back)->lane);
        }
        assert(neighbors[scenario->getNumCars() +car_id ] != nullptr && neighbors[scenario->getNumCars() +car_id]->id == neig.back);
    }

    if (neig.front == (size_t )-1) {
        assert(neighbors[car_id] == nullptr);
    } else {
        assert(neighbors[car_id] != nullptr && neighbors[car_id]->id == neig.front);
    }
}

__global__ void test_neighborsKernel(CudaScenario_id *scenario, TrafficObject_id ** dev_left_neighbors,TrafficObject_id ** dev_own_neighbors,TrafficObject_id ** dev_right_neighbors) {
    test_right_lane_neighbors(dev_right_neighbors, scenario);
    test_own_lane_neighbors(dev_own_neighbors, scenario);
    test_left_lane_neighbors(dev_left_neighbors, scenario);
}

__global__ void kernel_get_changes(Car_id::AdvanceData *change, CudaScenario_id * scenario_data,
                                   TrafficObject_id **right_lane_neighbors, TrafficObject_id **own_lane_neighbors, TrafficObject_id **left_lane_neighbors) {
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    CUDA_GLOBAL_ITER(car_idx, scenario_data->getNumCars()) {
        Lane_id::NeighboringObjectsRef rightNeighbors(right_lane_neighbors[car_idx], right_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        Lane_id::NeighboringObjectsRef ownNeighbors(own_lane_neighbors[car_idx], own_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        Lane_id::NeighboringObjectsRef leftNeighbors(left_lane_neighbors[car_idx], left_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        change[car_idx] = algorithm_wrapper.nextStep(*scenario_data->getCar(car_idx), leftNeighbors, ownNeighbors, rightNeighbors);
#ifdef RUN_WITH_TESTS
        if(car_idx == CAR_TO_ANALYZE) {
            printf("Changes for car(%lu) with Neighbors l(%lu, %lu), o(%lu, %lu), r(%lu, %lu): (%d, %f)\n", car_idx,
                   left_lane_neighbors[car_idx + scenario_data->getNumCars()] == nullptr ? (size_t )-1 : left_lane_neighbors[car_idx + scenario_data->getNumCars()]->id,
                   left_lane_neighbors[car_idx] == nullptr ? (size_t )-1 : left_lane_neighbors[car_idx]->id,
                   own_lane_neighbors[car_idx + scenario_data->getNumCars()] == nullptr ? (size_t )-1 : own_lane_neighbors[car_idx + scenario_data->getNumCars()]->id,
                   own_lane_neighbors[car_idx] == nullptr ? (size_t )-1 : own_lane_neighbors[car_idx]->id,
                   right_lane_neighbors[car_idx + scenario_data->getNumCars()] == nullptr ? (size_t )-1 : right_lane_neighbors[car_idx + scenario_data->getNumCars()]->id,
                   right_lane_neighbors[car_idx] == nullptr ? (size_t )-1 : right_lane_neighbors[car_idx]->id,
                   change[car_idx].lane_offset, change[car_idx].acceleration);
        }
#endif
    }
}


__global__ void updateSignalsKernel(CudaScenario_id * scenario_data) {
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    CUDA_GLOBAL_ITER(jnt_idx, scenario_data->getNumJunctions()) {
        algorithm_wrapper.updateSignals(*scenario_data->getJunction(jnt_idx));
    }
}


__global__ void testChangesKernel(CudaScenario_id *scenario_data, Car_id::AdvanceData *device_changes) {
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    for(size_t car_idx = GetGlobalIdx(); car_idx < scenario_data->getNumCars(); car_idx += GetGlobalDim()) {
        if (car_idx == 0) printf("testKernelChanges\n");
        Car_id::AdvanceData change = algorithm_wrapper.nextStep(*scenario_data->getCar(car_idx));
        if (!(change.lane_offset == device_changes[car_idx].lane_offset &&
              change.acceleration == device_changes[car_idx].acceleration)) {
            printf("Wrong change on lane(%7lu) - expected: (%5lu, %d, %.2f) got: (%lu, %d, %.2f)\n",
                   scenario_data->getCar(change.car)->lane, change.car, change.lane_offset,
                   change.acceleration, device_changes[car_idx].car, device_changes[car_idx].lane_offset,
                   device_changes[car_idx].acceleration);

        }
        assert(change.car == device_changes[car_idx].car);
        assert(change.lane_offset == device_changes[car_idx].lane_offset);
        assert(change.acceleration == device_changes[car_idx].acceleration);
    }
}


__global__ void applyChangesKernel(Car_id::AdvanceData *change, CudaScenario_id * scenario_data) {
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    CUDA_GLOBAL_ITER(car_idx, scenario_data->getNumCars()) {
        algorithm_wrapper.advanceStep(*scenario_data->getCar(change[car_idx].car), change[car_idx]);
    }
}

__global__ void testBucketsForInvalidLaneKernel(SortedBucketContainer *container, CudaScenario_id *scenario) {
    size_t bucket_id = GetBlockIdx();
    size_t object_idx = GetThreadIdx();
    TrafficObject_id::Cmp cmp;

    if(bucket_id == 0 && object_idx == 0) printf("container validity check\n");

    if (bucket_id < container->bucket_count) {
        auto &bucket = container->buckets[bucket_id];
        if (object_idx < bucket.size) {
            TrafficObject_id *object = bucket.buffer[object_idx];
            if(object == nullptr) printf("Wrong length in %lu\n", bucket_id);
            assert(object != nullptr);
            if(object->lane != bucket_id) {
                printf("Car(%lu) is not in Bucket(%lu)\n", object->id, object->lane);
                printf("Bucket(%lu) contents: ", container->buckets[object->lane].id);
                for (int i = 0; i < container->buckets[object->lane].size; i++) {
                    TrafficObject_id *p_obj = container->buckets[object->lane].buffer[i];
                    printf(" %lu(%.2f), ", p_obj == nullptr ? (size_t )-1 : p_obj->id, p_obj == nullptr ? -1. : p_obj->x);
                }
                printf("\n");
                printf("Bucket(%lu) contents: ", bucket.id);
                for (int i = 0; i < bucket.size; i++) {
                    TrafficObject_id *p_obj = bucket.buffer[i];
                    printf(" %lu(%.2f), ", p_obj == nullptr ? (size_t )-1 : p_obj->id, p_obj == nullptr ? -1. : p_obj->x);
                }
                printf("\n");
            }
            assert(object->lane == bucket_id);
            if (object_idx > 0) {
                if (!cmp(bucket.buffer[object_idx - 1], object)) {
                    printf("Sorting error in Lane(%lu)\n", bucket_id);
                }
                assert(cmp(bucket.buffer[object_idx - 1], object));
            }
        }
    }

    if (scenario != nullptr) {
        size_t car_idx = GetGlobalIdx();
        if (car_idx < scenario->getNumCars()) {
            auto &c = *scenario->getCar(car_idx);
            assert(c.lane < scenario->getNumLanes());
            auto &b = container->buckets[c.lane];
            bool found = false;
            for (int i = 0; i < b.size; i++) {
                assert(b.buffer[i] != nullptr);
                if (b.buffer[i]->id == c.id) {
                    found = true;
                    break;
                }
            }
            if (!found) printf("Car(%lu) not in container\n", car_idx);
            assert(found);
        }
    }
}

__device__ void find_nearest_for_car_on_lane(CudaScenario_id *scenario, SortedBucketContainer *container, TrafficObject_id &car, TrafficObject_id *&front, TrafficObject_id *&back) {

    BucketData &bucket = container->buckets[car.lane];
    TrafficObject_id::Cmp cmp;
    TrafficObject_id **traffic_object = upper_bound<TrafficObject_id*>(bucket.buffer, bucket.buffer + bucket.size, &car, cmp);
    assert(traffic_object <= bucket.buffer + bucket.size && traffic_object >= bucket.buffer);
    assert(bucket.size <= bucket.buffer_size);
#ifdef DEBUG_MSGS
    if (car.id == CAR_TO_ANALYZE) printf("Lane(%lu) has %lu objects\n", car.lane, bucket.size);
#endif
    if(bucket.buffer + bucket.size != traffic_object) {
        front = *traffic_object;
#ifdef DEBUG_MSGS
        if (car.id == CAR_TO_ANALYZE) printf("Car(%lu, %lu) front: Car(%lu)\n", car.id, car.lane, (*traffic_object)->id);
#endif
    } else {
#ifdef DEBUG_MSGS
        if (car.id == CAR_TO_ANALYZE) printf("Car(%lu, %lu) has clear heading.\n", car.id, car.lane);
#endif
        front = nullptr;
    }
    do {
        traffic_object--;
    } while(traffic_object > bucket.buffer && **traffic_object >= car);

    assert(traffic_object <= bucket.buffer + bucket.size && traffic_object >= bucket.buffer - 1);

    if(traffic_object >= bucket.buffer && bucket.buffer + bucket.size != traffic_object && **traffic_object < car) {
#ifdef DEBUG_MSGS
        if (car.id == CAR_TO_ANALYZE) printf("Car(%lu, %lu) back: Car(%lu)\n",  car.id, car.lane, (*traffic_object)->id);
#endif
        back = *traffic_object;
    } else {
#ifdef DEBUG_MSGS
        if (car.id == CAR_TO_ANALYZE) printf("Car(%lu, %lu) has clear back.\n", car.id, car.lane);
#endif
        back = nullptr;
    }
}


__global__ void find_nearest2(CudaScenario_id *scenario, SortedBucketContainer *container, TrafficObject_id **nearest_left,
                              TrafficObject_id **nearest_own, TrafficObject_id **nearest_right) {

    AlgorithmWrapper wrapper(*scenario);
    CUDA_GLOBAL_ITER(idx, scenario->getNumCars() * 3) {
        int lane_offset = (int) (idx % 3) - 1;
        size_t car_idx = idx / 3;

        TrafficObject_id car = *scenario->getCar(car_idx);
        assert(scenario->getCar(car_idx) != nullptr);

        size_t lane_id = (size_t) -1;
        TrafficObject_id **nearest = nullptr;
        Road_id::NeighboringLanes n_lanes;
        switch (lane_offset) {
            case 1:
                n_lanes = wrapper.getNeighboringLanes(*scenario->getLane(car.lane));
                lane_id = n_lanes.right;
                nearest = nearest_right;
                break;
            case 0:
                lane_id = car.lane;
                nearest = nearest_own;
                break;
            case -1:
                n_lanes = wrapper.getNeighboringLanes(*scenario->getLane(car.lane));
                lane_id = n_lanes.left;
                nearest = nearest_left;
                break;
            default:
                assert(false);
        }
#ifdef DEBUG_MSGS
        if (car.id == CAR_TO_ANALYZE) printf("original lane of Car(%lu) is Lane(%lu)\n", car.id, car.lane);
#endif
        car.lane = lane_id;
        TrafficObject_id *&nearest_font = nearest[car_idx];
        TrafficObject_id *&nearest_back = nearest[car_idx + scenario->getNumCars()];

        if (lane_id == (size_t) -1) {
            nearest_back = nullptr;
            nearest_font = nullptr;
            continue;
        }

        assert(lane_id < container->bucket_count);
        if (container->buckets[lane_id].size == 0) {
            nearest_back = nullptr;
            nearest_font = nullptr;
        } else {
            find_nearest_for_car_on_lane(scenario, container, car, nearest_font, nearest_back);
        }

        Lane_id *l = scenario->getLane(car.lane);
        assert(l != nullptr);
        RedTrafficLight_id *tl = scenario->getLight(l->traffic_light);
        assert(tl != nullptr);
        if (tl->isRed()) {
            if (car < *tl && (nearest_font == nullptr || *tl < *nearest_font)) {
                nearest_font = tl;
#ifdef DEBUG_MSGS
                if (car.id == CAR_TO_ANALYZE)
                    printf("Car(%lu, %lu) has light(%lu, %lu) in the front.\n", car.id, car.lane, nearest_font->id,
                           nearest_font->lane);
#endif
            }
            /**
            if (car > *tl && (nearest_back == nullptr || *tl > *nearest_back)) {
                // nearest_back = tl;
#ifdef DEBUG_MSGS
                if (car.id == CAR_TO_ANALYZE)
                    printf("Car(%lu, %lu) has light(%lu, %lu) in the back.\n", car.id, car.lane, nearest_back->id,
                           nearest_back->lane);
#endif
            }*/
        }

    }
}

void static_advance(size_t steps, Scenario_id &scenario) {

    SortBuffer preSumBuffer(scenario, SUGGESTED_THREADS);
    size_t number_of_lanes = scenario.lanes.size();
    size_t number_of_cars = scenario.cars.size();

    CudaScenario_id *device_cuda_scenario = CudaScenario_id::fromScenarioData_device(scenario);
    std::shared_ptr<SortedBucketContainer> bucket_memory = SortedBucketContainer::fromScenario(scenario, device_cuda_scenario, preSumBuffer);

    TrafficObject_id **dev_left_neighbors, **dev_own_neighbors, **dev_right_neighbors;
    GPU_ALLOC((void **) &dev_left_neighbors, 2 * number_of_cars * sizeof(TrafficObject_id *))
    GPU_ALLOC((void **) &dev_own_neighbors, 2 * number_of_cars * sizeof(TrafficObject_id *))
    GPU_ALLOC((void **) &dev_right_neighbors, 2 * number_of_cars * sizeof(TrafficObject_id *))

    Car_id::AdvanceData *device_changes;
    GPU_ALLOC((void **) &device_changes, scenario.cars.size() * sizeof(Car_id::AdvanceData))

#ifdef DEBUG_MSGS
    printf("Starting to advance scenario...\n\n");
#endif
#ifdef RUN_WITH_TESTS
    if(CAR_TO_ANALYZE < scenario.cars.size())
        printf("Car(%lu) on Lane(%lu)\n", (size_t) CAR_TO_ANALYZE, scenario.cars[CAR_TO_ANALYZE].lane);
#endif
    for (int i = 0; i < steps; i++) {
#ifdef DEBUG_MSGS
#endif
        SortedBucketContainer::RestoreValidState(scenario, bucket_memory.get(), preSumBuffer);

#ifdef RUN_WITH_TESTS
        testBucketsForInvalidLaneKernel<<<number_of_lanes, 1024>>>(bucket_memory.get(), device_cuda_scenario);
        CHECK_FOR_ERROR();
#endif

        find_nearest2<<<scenario.cars.size() / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>(device_cuda_scenario, bucket_memory.get(), dev_left_neighbors, dev_own_neighbors, dev_right_neighbors);
        CHECK_FOR_ERROR();

#ifdef RUN_WITH_TESTS
        test_neighborsKernel<<<number_of_cars / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>
            (device_cuda_scenario, dev_left_neighbors, dev_own_neighbors, dev_right_neighbors);
        CHECK_FOR_ERROR();
#endif

        kernel_get_changes<<<number_of_cars / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>
            (device_changes, device_cuda_scenario, dev_right_neighbors, dev_own_neighbors, dev_left_neighbors);
        CHECK_FOR_ERROR();

#ifdef RUN_WITH_TESTS
        testChangesKernel<<<number_of_cars / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>
            (device_cuda_scenario, device_changes);
        CHECK_FOR_ERROR();
#endif

        applyChangesKernel<<<number_of_cars / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>(device_changes, device_cuda_scenario);
        CHECK_FOR_ERROR();

        updateSignalsKernel<<<scenario.junctions.size() / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>(device_cuda_scenario);
        CHECK_FOR_ERROR();

    }

    device_cuda_scenario->retriveData(&scenario);

    gpuErrchk(cudaFree(dev_left_neighbors));
    gpuErrchk(cudaFree(dev_own_neighbors));
    gpuErrchk(cudaFree(dev_right_neighbors));
    gpuErrchk(cudaFree(device_changes));
}

void CudaAlgorithm::advance(size_t steps) {
    static_advance(steps, *getIDScenario());
    cudaDeviceReset();
};