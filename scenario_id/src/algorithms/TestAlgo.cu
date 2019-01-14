//
// Created by oke on 09.01.19.
//

#include <curand_mtgp32_kernel.h>
#include <model/Lane.h>
#include "algorithms/TestAlgo.h"
#include "cuda/cuda_utils.h"

#define BUCKET_TO_ANALYZE 5
#define CAR_TO_ANALYZE 0 // 308


#define SUGGESTED_THREADS 512


__device__
size_t GetGlobalIdx(){
    return + blockIdx.z * blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y
           + blockIdx.y * blockDim.x * blockDim.y * blockDim.z * gridDim.x
           + blockIdx.x * blockDim.x * blockDim.y * blockDim.z
           + threadIdx.z * blockDim.y * blockDim.x
           + threadIdx.y * blockDim.x
           + threadIdx.x;
}
__device__
size_t GetThreadIdx(){
    return threadIdx.z * blockDim.y * blockDim.x
           + threadIdx.y * blockDim.x
           + threadIdx.x;
}
__device__
size_t GetBlockIdx(){
    return blockIdx.z * gridDim.y * gridDim.x
           + blockIdx.y * gridDim.x
           + blockIdx.x;
}

__device__
size_t GetGridDim(){
    return gridDim.y * gridDim.x * gridDim.z;
}

__device__
size_t GetBlockDim(){
    return blockDim.y * blockDim.x * blockDim.z;
}

__device__
size_t GetGlobalDim(){
    return blockDim.y * blockDim.x * blockDim.z * gridDim.y * gridDim.x * gridDim.z;
}



__global__ void FixSizeKernel(BucketMemory *container, bool only_lower) {

    for (size_t lane_id = GetBlockIdx(); lane_id < container->bucket_count; lane_id += GetGridDim()) {
        auto &bucket = container->buckets[lane_id];
        bool found = false;
        size_t new_size;
        size_t max_size = only_lower ? bucket.size : bucket.buffer_size;
        for (size_t idx = GetThreadIdx(); idx < bucket.buffer_size; idx += GetBlockDim()) {
            if (idx == 0) {
                if (bucket.buffer[0] == nullptr) {
                    new_size = 0;
                    found = true;
                }
            } else {
                if (bucket.buffer[idx] == nullptr && bucket.buffer[idx - 1] != nullptr) {
                    new_size = idx;
                    found = true;
                }
            }
        }

        if (found)
            bucket.size = new_size;

#ifdef RUN_WITH_TESTS
        if (found) {
            if(lane_id == BUCKET_TO_ANALYZE)
                printf("Final Bucket(%lu) size: %lu\n", lane_id, new_size);
        }
#endif
    }
}

__global__ void bucketMemoryInitializeKernel(BucketMemory *bucketmem,  BucketData *buckets, TrafficObject_id **main_buffer, CudaScenario_id *cuda_device_scenario, float bucket_memory_factor) {
    new(bucketmem)BucketMemory(cuda_device_scenario, buckets, main_buffer, bucket_memory_factor);
}

__global__ void bucketMemoryLoadKernel(BucketMemory *bucketmem, CudaScenario_id *cuda_device_scenario, float bucket_memory_factor) {
    size_t lane_idx = GetGlobalIdx();

    if (lane_idx >= bucketmem->bucket_count) return;

    BucketData &bucket = bucketmem->buckets[lane_idx];

    for(auto &c : cuda_device_scenario->getCarIterator()) {
        if(lane_idx == c.lane) {
            assert(bucket.size + 1 < bucket.buffer_size);
            bucket.size += 1;
            bucket.buffer[bucket.size - 1] = &c;
        }
    }
    //printf("%lu, %lu ---- %lu/%lu: %lu\n", (size_t )threadIdx.x, (size_t )blockIdx.x, lane_idx, device_bucketContainer->getNumBuckets(), bucket.getSize());
}

__global__
void bucketMemoryLoadKernel2(BucketMemory *bucketmem, CudaScenario_id *cuda_device_scenario, unsigned int *temp_value) {
    for(size_t car_idx = GetGlobalIdx(); car_idx < cuda_device_scenario->getNumCars(); car_idx += GetGlobalDim()) {
        auto car = cuda_device_scenario->getCar(car_idx);
        size_t insert_offset = atomicAdd(temp_value + car->lane, 1);
        bucketmem->buckets[car->lane].buffer[insert_offset] = car;
        // printf("Insert: %lu at Lane(%lu) at %lu\n", car_idx, car->lane, insert_offset);
    }
}

struct free_deleter
{
    void operator()(void* m) {
        BucketMemory *memory = (BucketMemory *) m;
        BucketMemory memory_host;

        gpuErrchk(cudaMemcpy(&memory_host, memory, sizeof(BucketMemory), cudaMemcpyDeviceToHost));

        gpuErrchk(cudaFree(memory_host.main_buffer));
        gpuErrchk(cudaFree(memory_host.buckets));
        gpuErrchk(cudaFree(memory));
    }
};

CUDA_HOSTDEV size_t BucketMemory::getBufferSize(CudaScenario_id &scenario, float bucket_memory_factor) {
    size_t total_buffer_size = 0;
    for (Lane_id &l : scenario.getLaneIterator()) {
        total_buffer_size += ceil(bucket_memory_factor * scenario.getRoad(l.road)->length / 5.);
    }
    return total_buffer_size;
}


CUDA_HOST size_t BucketMemory::getBufferSize(Scenario_id &scenario, float bucket_memory_factor) {
    size_t total_buffer_size = 0;
    for (Lane_id &l : scenario.lanes) {
        total_buffer_size += ceil(bucket_memory_factor * scenario.roads.at(l.road).length / 5.);
    }
    return total_buffer_size;
}


CUDA_DEV BucketMemory::BucketMemory(CudaScenario_id *scenario, BucketData *_buckets, TrafficObject_id **_main_buffer, float bucket_memory_factor) {

    bucket_count = scenario->getNumLanes();

    buckets = _buckets; // new BucketData[scenario->getNumLanes()];
    main_buffer = _main_buffer; // new TrafficObject_id*[total_buffer_size];
    assert(main_buffer != nullptr);
    assert(buckets != nullptr);

    size_t total_buffer_size = 0;
    size_t i = 0;
    for (Lane_id &l : scenario->getLaneIterator()) {
        buckets[i].size = 0;
        buckets[i].buffer_size = ceil(bucket_memory_factor * scenario->getRoad(l.road)->length / 5.);
        total_buffer_size += buckets[i].buffer_size;
        buckets[i].buffer = main_buffer + total_buffer_size;
        i++;
    }

    this->main_buffer_size = total_buffer_size;

    printf("Allocated: %.2fMB\n", (float) (sizeof(size_t) * scenario->getNumLanes() * 2 + sizeof(TrafficObject_id**) * scenario->getNumLanes() +
                                           sizeof(TrafficObject_id*) * total_buffer_size) / 1024. / 1024.);
}


CUDA_HOST void BucketMemory::test(BucketMemory *memory) {

    BucketMemory memory_host;
    gpuErrchk(cudaMemcpy(&memory_host, memory, sizeof(BucketMemory), cudaMemcpyDeviceToHost));

    printf("BucketMemory: %lu Buckets\n", memory_host.bucket_count);

}

std::shared_ptr<BucketMemory> BucketMemory::fromScenario(Scenario_id &scenario, CudaScenario_id *device_cuda_scenario) {
    TrafficObject_id **main_buffer;
    gpuErrchk(cudaMalloc((void**) &main_buffer, BucketMemory::getBufferSize(scenario, 4) * sizeof(TrafficObject_id*)));

    BucketData *buckets;
    gpuErrchk(cudaMalloc((void**) &buckets, scenario.lanes.size() * sizeof(BucketData)));

    // allocate bucket class
    BucketMemory *bucket_memory;
    gpuErrchk(cudaMalloc((void**) &bucket_memory, sizeof(BucketMemory)));

    // initialize bucket class
    bucketMemoryInitializeKernel<<<1, 1>>>(bucket_memory, buckets, main_buffer, device_cuda_scenario, 4.);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // load data into memory
    // bucketMemoryLoadKernel<<<1024, 1024>>>(bucket_memory, device_cuda_scenario, 4.);

    unsigned int *tmp;
    gpuErrchk(cudaMalloc((void**) &tmp, scenario.lanes.size() * sizeof(unsigned int )));
    gpuErrchk(cudaMemset(tmp, 0, scenario.lanes.size() * sizeof(unsigned int )));
    bucketMemoryLoadKernel2<<<1024, 1024>>>(bucket_memory, device_cuda_scenario, tmp);
    gpuErrchk(cudaFree(tmp));

    FixSizeKernel<<<256, 20>>>(bucket_memory, false);

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    std::shared_ptr<BucketMemory> result(bucket_memory, free_deleter());
    return result;
}

template<typename T>
__device__ void cuda_swap(T &t1, T &t2) {
    T t = t1;
    t1 = t2;
    t2 = t;
}


template<typename T>
__global__ void bitonic_sort_step_kernel(T *dev_values, int j, int k, int n) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = GetGlobalIdx();
    ixj = i ^ j;

    if (i >= n || ixj >= n)
        return;
    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (dev_values[i] > dev_values[ixj]) {
                /* exchange(i,ixj); */
                cuda_swap(dev_values[i], dev_values[ixj]);
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (dev_values[i] < dev_values[ixj]) {
                /* exchange(i,ixj); */
                cuda_swap(dev_values[i], dev_values[ixj]);
            }
        }
    }
}


template<typename T>
__global__ void bitonic_sort_merge_kernel(T* values, int k, int n) {
    unsigned int i; /* Sorting partners: i and ixj */
    i = GetGlobalIdx();
    if(i + k < n && values[i] > values[i + k])
        cuda_swap(values[i], values[i + k]);
}

#define THREADS 512 // 2^9
template <typename T>
void dev_mem_bitonic_sort(T *device_values, unsigned long n) {
    unsigned long block_num = (unsigned int) ceil(n / (float) THREADS);
    unsigned long block_num2 = 1;
    // printf("%d Threads on %lux%lu Blocks\n", THREADS, block_num, block_num2);
    if (block_num > 65535) {
        block_num2 = 65535;
        block_num = (int) ceil((float) block_num / (float) block_num2);
    }
    dim3 blocks(block_num, block_num2);    /* Number of blocks   */
    dim3 threads(THREADS, 1);  /* Number of threads  */
    int j, k;
    /* Major step */
    for (k = 2; k <= n; k <<= 1) {
        /* Minor step */
        for (j=k>>1; j>0; j=j>>1) {
            bitonic_sort_step_kernel<<<blocks, threads>>>(device_values, j, k, n);
            gpuErrchk( cudaPeekAtLastError() );
        }
    }

    unsigned long power = pow(2, floor(log(n)/log(2)));
    for (unsigned long k = power; k > 0; k >>= 1) {
        bitonic_sort_merge_kernel<<<blocks, threads>>>(device_values, k, n);
        gpuErrchk( cudaPeekAtLastError() );
    }
}

template<typename T, typename Cmp>
__device__ void bitonic_sort_step(T *dev_values, int j, int k, int n, Cmp cmp) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * threadIdx.y;
    ixj = i ^ j;

    if (i >= n || ixj >= n)
        return;
    /* The threads with the lowest ids sort the array. */
    if ((ixj) > i) {
        if ((i & k) == 0) {
            /* Sort ascending */
            if (cmp(dev_values[ixj], dev_values[i])) {
                /* exchange(i,ixj); */
                cuda_swap(dev_values[i], dev_values[ixj]);
            }
        }
        if ((i & k) != 0) {
            /* Sort descending */
            if (cmp(dev_values[i], dev_values[ixj])) {
                /* exchange(i,ixj); */
                cuda_swap(dev_values[i], dev_values[ixj]);
            }
        }
    }
}


template<typename T, typename Cmp>
__device__ void bitonic_sort_merge(T* values, int k, int n, Cmp cmp) {
    unsigned int i; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * threadIdx.y;
    if(i + k < n && cmp(values[i + k], values[i]))
        cuda_swap(values[i], values[i + k]);
}

__device__ __host__ uint64_t next_pow2m1(uint64_t x) {
    x--;
    x |= x>>1;
    x |= x>>2;
    x |= x>>4;
    x |= x>>8;
    x |= x>>16;
    x |= x>>32;
    x++;
    return x;
}

template <typename Cmp>
__global__ void cudaSortKernel(BucketMemory *container, Cmp cmp) {
    size_t first_bucket_idx = GetBlockIdx();
    size_t i = GetThreadIdx();
    size_t block_count = GetGridDim();

    for(size_t bucket_idx = first_bucket_idx; bucket_idx < container->bucket_count; bucket_idx += block_count) {
#ifdef RUN_WITH_TESTS
        if(i == 0 && bucket_idx == BUCKET_TO_ANALYZE) {
            printf("Bucket(%lu) contents before sorting: ", bucket_idx);
            for (int i = 0; i < container->buckets[bucket_idx].size; i++) {
                TrafficObject_id *p_obj = container->buckets[bucket_idx].buffer[i];
                printf(" %lu(%.2f), ", p_obj == nullptr ? (size_t )-1 : p_obj->id, p_obj == nullptr ? -1. : p_obj->x);
            }
            printf("\n");
        }
#endif
        assert(container->buckets[bucket_idx].size < GetBlockDim());

        if (container->buckets[bucket_idx].size > 1) {
            TrafficObject_id **device_values = container->buckets[bucket_idx].buffer;
            size_t n = container->buckets[bucket_idx].size;
            // printf("bucket %lu with %lu items.\n", bucket_idx, n);
            int j, k;
            unsigned long power = next_pow2m1(n);
            assert(power >= n);
            /* Major step */
            for (k = 2; k <= power; k <<= 1) {
                /* Minor step */
                for (j = k >> 1; j > 0; j = j >> 1) {
                    bitonic_sort_step(device_values, j, k, n, cmp);
                    __syncthreads();
                }
            }

            for (unsigned long k = power; k > 0; k >>= 1) {
                bitonic_sort_merge(device_values, k, n, cmp);
                __syncthreads();
            }
        }

#ifdef RUN_WITH_TESTS
        if(i == 0 && bucket_idx == BUCKET_TO_ANALYZE) {
            printf("Bucket(%lu) contents after sorting: ", bucket_idx);
            for (int i = 0; i < container->buckets[bucket_idx].size; i++) {
                TrafficObject_id *p_obj = container->buckets[bucket_idx].buffer[i];
                printf(" %lu(%.2f), ", p_obj == nullptr ? (size_t )-1 : p_obj->id, p_obj == nullptr ? -1. : p_obj->x);
            }
            printf("\n");
        }
#endif

    }
}

__global__ void find_nearest(CudaScenario_id *scenario, BucketMemory *container, TrafficObject_id **nearest_left,
        TrafficObject_id **nearest_own, TrafficObject_id **nearest_right) {
    AlgorithmWrapper wrapper(*scenario);

    size_t car_idx = GetGlobalIdx();
    int lane_offset = (int)(car_idx % 3) - 1;
    car_idx /= 3;

    if (car_idx >= scenario->getNumCars())
        return;

    TrafficObject_id car = *scenario->getCar(car_idx);
    size_t lane_id = (size_t ) -1;
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
    car.lane = lane_id;
    size_t n = container->buckets[lane_id].size;
    TrafficObject_id *&nearest_font = nearest[car_idx];
    TrafficObject_id *&nearest_back = nearest[car_idx + scenario->getNumCars()];

    if (lane_id == (size_t) -1) {
        nearest_back = nullptr;
        nearest_font = nullptr;
        return;
    }
    if (n == 0) {
        nearest_back = nullptr;
        nearest_font = nullptr;
    } else {

        TrafficObject_id **lane_objects = container->buckets[lane_id].buffer;
        size_t search_idx = n / 2;
        size_t from = 0;
        size_t to = n;

        if (n == 1) {
            if (*lane_objects[0] == car) {
                nearest_font = nullptr;
                nearest_back = nullptr;
            } else if (*lane_objects[0] > car) {
                nearest_font = lane_objects[0];
                nearest_back = nullptr;
            } else if (*lane_objects[0] < car) {
                nearest_back = lane_objects[0];
                nearest_font = nullptr;
            }
        } else {
            while (true) {
#ifdef DEBUG_MSGS
                if (car.id == CAR_TO_ANALYZE)
                    printf("Find (%lu/%lu) on Lane(%lu): %lu/%.2f, Current(%lu) %lu/%.2f, Index: %lu/%lu/%lu \n",
                           car.id, scenario->getCar(car_idx)->lane, lane_id, car.lane, car.x,
                           lane_objects[search_idx]->id, lane_objects[search_idx]->lane, lane_objects[search_idx]->x,
                           from,
                           search_idx, to);
#endif
                if (*lane_objects[search_idx] < car) {
                    if (search_idx + 1 == n || *lane_objects[search_idx + 1] >= car) {
                        break;
                    }
                    from = search_idx + 1;
                    search_idx += (to - from) / 4 == 0 ? 1 : (to - from) / 4;
                } else {
                    to = search_idx;
                    search_idx -= (to - from) / 4 == 0 ? 1 : (to - from) / 4;
                }
                if ((to - from) == 1)
                    break;
            }

#ifdef DEBUG_MSGS
            if (car.id == CAR_TO_ANALYZE)
                printf("Find (%lu/%lu) on Lane(%lu): %lu/%.2f, Current(%lu) %lu/%.2f, Index: %lu/%lu/%lu \n",
                        car.id, scenario->getCar(car_idx)->lane, lane_id, car.lane, car.x,
                        lane_objects[search_idx]->id, lane_objects[search_idx]->lane, lane_objects[search_idx]->x, from,
                        search_idx, to);
#endif

            assert(search_idx < n && (*lane_objects[search_idx] < car || search_idx == 0));

            if (search_idx == 0 && *lane_objects[search_idx] >= car) {
                nearest_back = nullptr;
                while (search_idx < n && *lane_objects[search_idx] == car) search_idx++;
                if (search_idx < n)
                    nearest_font = lane_objects[search_idx];
                else
                    nearest_font = nullptr;
            } else {
                if (search_idx < n)
                    nearest_back = lane_objects[search_idx];
                else
                    nearest_back = nullptr;
                search_idx++;
                while (search_idx < n && *lane_objects[search_idx] == car) search_idx++;
                if (search_idx < n) {
                    nearest_font = lane_objects[search_idx];
                } else {
                    nearest_font = nullptr;
                }
            }
        }
    }

#ifdef DEBUG_MSGS
    if(car.id == CAR_TO_ANALYZE) {
        printf("Found(%lu/%lu) on Lane(%lu): %lu %lu\n", car.id, scenario->getCar(car_idx)->lane, lane_id,
               (nearest_back != nullptr ? nearest_back->id : (size_t) -1),
               nearest_font != nullptr ? nearest_font->id : (size_t) -1);
    }
#endif

    Lane_id *l = scenario->getLane(car.lane);
    RedTrafficLight_id *tl = scenario->getLight(l->traffic_light);
    if(tl->isRed()) {
        if (car < *tl && (nearest_font == nullptr || *tl < *nearest_font)) {
            nearest_font = tl;
        }
        if (car > *tl && (nearest_back == nullptr || *tl > *nearest_back)) {
            nearest_back = tl;
        }
    }


}

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
            printf("c - %5lu, FrontCar(%5lu) at Lane(%lu)\n", car_id, neig.back, scenario->getCar(neig.back)->lane);
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
    size_t car_idx = GetGlobalIdx();
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    if (car_idx < scenario_data->getNumCars()) {
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
    size_t jnt_idx = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    if (jnt_idx < scenario_data->getNumJunctions()) {
        algorithm_wrapper.updateSignals(*scenario_data->getJunction(jnt_idx));
    }
}


__global__ void testChangesKernel(CudaScenario_id *scenario_data, Car_id::AdvanceData *device_changes) {
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    size_t car_idx = GetGlobalIdx();
    if(car_idx == 0) printf("testKernelChanges\n");
    if (car_idx >= scenario_data->getNumCars()) return;
    Car_id::AdvanceData change = algorithm_wrapper.nextStep(*scenario_data->getCar(car_idx));
    if(!(change.lane_offset == device_changes[car_idx].lane_offset &&
            change.acceleration == device_changes[car_idx].acceleration)) {
        printf("Wrong change on lane(%7lu) - expected: (%5lu, %d, %.2f) got: (%lu, %d, %.2f)\n", scenario_data->getCar(change.car)->lane, change.car, change.lane_offset,
                change.acceleration, device_changes[car_idx].car, device_changes[car_idx].lane_offset,
                device_changes[car_idx].acceleration);

    }
    assert(change.car == device_changes[car_idx].car);
    assert(change.lane_offset == device_changes[car_idx].lane_offset);
    assert(change.acceleration == device_changes[car_idx].acceleration);
}


__global__ void applyChangesKernel(Car_id::AdvanceData *change, CudaScenario_id * scenario_data) {
    size_t car_idx = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    if (car_idx < scenario_data->getNumCars()) {
        algorithm_wrapper.advanceStep(*scenario_data->getCar(change[car_idx].car), change[car_idx]);
    }
}

__global__ void testBucketsForInvalidLaneKernel(BucketMemory *container, CudaScenario_id *scenario) {
    size_t bucket_id = GetBlockIdx();
    size_t object_idx = GetThreadIdx();
    TrafficObject_id::Cmp cmp;

    if(bucket_id == 0 && object_idx == 0) printf("container validity check\n");

    if (bucket_id < container->bucket_count) {
        auto &bucket = container->buckets[bucket_id];
        if (object_idx < bucket.size) {
            TrafficObject_id *object = bucket.buffer[object_idx];
            assert(object != nullptr);
            assert(object->lane == bucket_id);
            if (object_idx > 0) {
                if (!cmp(bucket.buffer[object_idx - 1], object)) {
                    printf("Sorting error in Lane(%lu)\n", bucket_id);
                }
                assert(cmp(bucket.buffer[object_idx - 1], object));
            }
        }
    }

    size_t car_idx = GetGlobalIdx();
    if(car_idx < scenario->getNumCars()) {
        auto &c = *scenario->getCar(car_idx);
        auto &b = container->buckets[c.lane];
        bool found = false;
        for (int i = 0; i < b.size; i++) {
            if (b.buffer[i]->id == c.id) {
                found = true;
                break;
            }
        }
        if (!found) printf("Car(%lu) not in container\n", car_idx);
        assert(found);
    }
}

__device__ inline bool isInWrongLane(BucketMemory *container, TrafficObject_id **object) {
    if (*object == nullptr) return false;
    BucketData supposed_bucket = container->buckets[(*object)->lane];
    return (object < supposed_bucket.buffer || supposed_bucket.buffer + supposed_bucket.size <= object);
}

CUDA_HOSTDEV inline bool IsPowerOfTwo(unsigned long x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}


__device__ void PreScan(size_t *temp, size_t idx, size_t n, size_t skip=1) {

    assert(IsPowerOfTwo(n));
    assert(IsPowerOfTwo(skip));
    if (!(2 * idx < n)) printf("%lu, %lu\n", 2*idx, n);
    assert(2 * idx < n);

    int offset = 1;

    n /= skip;

    for (int d = n >> 1; d > 0; d >>= 1) // build sum in place up the tree
    {
        __syncthreads();

        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;

            temp[bi] += temp[ai * skip];
        }
        offset *= 2;
    }
    size_t total_sum = 0;
    if (idx == 0) {
        total_sum = temp[n - 1];
        temp[n - 1] = 0;
    } // clear the last element

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
    {
        offset >>= 1;
        __syncthreads();

        if (idx < d) {
            int ai = offset * (2 * idx + 1) - 1;
            int bi = offset * (2 * idx + 2) - 1;
            //printf("%d, %d, %d, %d\n", ai, bi, offset, idx);

            size_t t = temp[ai * skip];
            temp[ai * skip] = temp[bi * skip];
            temp[bi * skip] += t;
        }
    }

    __syncthreads();
    size_t t1 = temp[2 * idx];
    size_t t2 = temp[2 * idx + 1];
    __syncthreads();

    if (idx != 0)
        temp[2 * idx - 1] = t1;
    else
        temp[n - 1] = total_sum;
    temp[2 * idx] = t2;
}

__global__ void BlockWisePreScan(BucketMemory *container, size_t *g_odata, size_t n) {
    assert(IsPowerOfTwo(n));
    extern __shared__ size_t temp[];// allocated on invocation

    size_t traffic_object_id = GetThreadIdx();
    size_t buffer_offset = GetBlockIdx() * n;
    assert(n == GetBlockDim() * 2);
    assert(GetGlobalDim() * 2 >= container->main_buffer_size);

    TrafficObject_id **p_obj1 = container->main_buffer + 2 * traffic_object_id + buffer_offset;
    TrafficObject_id **p_obj2 = container->main_buffer + 2 * traffic_object_id + 1 + buffer_offset;
    // if(GetGlobalIdx() == 0) printf("Starting...\n");

    if(p_obj1 < container->main_buffer + container->main_buffer_size && p_obj1 != nullptr) {
        temp[2 * traffic_object_id] = isInWrongLane(container, p_obj1) ? 1 : 0; // load input into shared memory
        //printf("Got one. %lu, %lu, %lu \n", traffic_object_id, buffer_offset, temp[2 * traffic_object_id]);
    } else
        temp[2 * traffic_object_id] = 0;

    if(p_obj2 < container->main_buffer + container->main_buffer_size && p_obj2 != nullptr) {
        temp[2 * traffic_object_id + 1] = isInWrongLane(container, p_obj2) ? 1 : 0;  // load input into shared memory
        //printf("Got one. %lu, %lu, %lu \n", traffic_object_id, buffer_offset, temp[2 * traffic_object_id + 1]);
    } else
        temp[2 * traffic_object_id + 1] = 0;

    PreScan(temp, traffic_object_id, n);

    __syncthreads();

    if(p_obj1 < container->main_buffer + container->main_buffer_size)
        g_odata[2 * traffic_object_id + buffer_offset] = temp[2*traffic_object_id]; // write results to device memory
    if(p_obj2 < container->main_buffer + container->main_buffer_size)
        g_odata[2 * traffic_object_id+1 + buffer_offset] = temp[2*traffic_object_id+1];

}

__global__ void collect_changedKernel(BucketMemory *container) {
    BucketData last_bucket = container->buckets[container->bucket_count - 1];
    TrafficObject_id **it = container->main_buffer + GetGlobalIdx();
    if(it < last_bucket.buffer + last_bucket.size) {
        if ((*it) == nullptr) return;
        BucketData supposed_bucket = container->buckets[(**it).lane];
        if(it < supposed_bucket.buffer || supposed_bucket.buffer + supposed_bucket.size <= it) {
            printf("%lu is in wrong lane (actual: %lu). (%lu) \n", (**it).id, (**it).lane, (it - container->main_buffer));
        }
    }

}

__global__ void MergeBlockWisePreScan(size_t *out, size_t out_size, size_t *in, size_t in_size, size_t buffer_size, size_t skip_count) {
    assert(IsPowerOfTwo(buffer_size));
    size_t idx = GetThreadIdx();
    size_t offset = GetBlockIdx() * buffer_size * skip_count;

    assert(GetBlockDim() * 2 == buffer_size);
    assert((2 * GetBlockDim() + 1) * skip_count - 1 + GetGridDim() * buffer_size * skip_count >= in_size);

    extern __shared__ size_t temp[];

    if ((2 * idx + 1) * skip_count - 1 + offset < in_size)
        temp[2 * idx] = in[(2 * idx + 1) * skip_count - 1 + offset];
    else
        temp[2 * idx] = 0;

    if ((2 * idx + 2) * skip_count - 1 + offset < in_size)
        temp[2 * idx + 1] = in[(2 * idx + 2) * skip_count - 1 + offset];
    else
        temp[2 * idx + 1] = 0;

    __syncthreads();

    PreScan(temp, idx, buffer_size);

    __syncthreads();

    assert(2 * GetBlockDim() + 1 + buffer_size * GetGridDim() >= out_size);
    if(2 * idx + buffer_size * GetBlockIdx() < out_size)
        out[2 * idx + buffer_size * GetBlockIdx()] = temp[2 * idx];
    if(2 * idx + 1 + buffer_size * GetBlockIdx() < out_size)
        out[2 * idx + 1 + buffer_size * GetBlockIdx()] = temp[2 * idx + 1];
}

__global__ void MergeBlockWisePreScanStep2(size_t *out, size_t *in, size_t n, size_t buffer_size, size_t out_size) {
    size_t idx = GetThreadIdx();
    size_t offset = GetBlockIdx() * n;

    if (GetBlockIdx() > 0 && idx + offset < out_size) {
        out[idx + offset] += in[GetBlockIdx() - 1];
    }

}

__global__ void MoveToReinsertBuffer(BucketMemory *container, size_t *prefixSum, size_t n,
        TrafficObject_id **reinsert_buffer, size_t buffer_size) {

    size_t idx = GetGlobalIdx();

    assert(GetGlobalDim() >= n);

    if (idx >= n) return;

    if ((idx == 0 && prefixSum[0] > 0) || (idx != 0 && prefixSum[idx] != prefixSum[idx - 1])){
        size_t insert_id = idx == 0 ? 0 : prefixSum[idx - 1];
        if(insert_id >= buffer_size) {
            printf("%lu, %lu\n", insert_id, buffer_size);
        }

        if(!(insert_id < buffer_size && container->main_buffer[idx] != nullptr)) {
            printf("%lu: %lu - %lu\n", idx, prefixSum[idx], idx == 0 ? -1 : prefixSum[idx - 1]);
        }

        assert(insert_id < buffer_size && container->main_buffer[idx] != nullptr);
        reinsert_buffer[insert_id] = container->main_buffer[idx];
        container->main_buffer[idx] = nullptr;
#ifdef RUN_WITH_TESTS
        if (reinsert_buffer[insert_id]->id == CAR_TO_ANALYZE) {
            printf("Car(%lu) moved to ReinsertBuffer(#%lu)\n", reinsert_buffer[insert_id]->id, insert_id);
        }
#endif
    }
}

__device__ inline void insert_into_bucket(size_t idx, BucketData &bucket, TrafficObject_id **reinsertBuffer,  size_t *insert_into_lane, size_t n, size_t lane_id, size_t offset) {

    if (idx + offset < n) {
        if ((idx == 0 && insert_into_lane[0] > 0) || (idx != 0 && insert_into_lane[idx] != insert_into_lane[idx - 1])) {
            size_t insert_id = idx == 0 ? 0 : insert_into_lane[idx - 1];
            assert(bucket.buffer[bucket.size + insert_id] == nullptr);
            bucket.buffer[bucket.size + insert_id] = reinsertBuffer[idx + offset];
#ifdef RUN_WITH_TESTS
            if (reinsertBuffer[idx + offset]->id == CAR_TO_ANALYZE) {
                printf("Car(%lu) from ReinsertBuffer(#%lu) to Bucket(%lu)\n", reinsertBuffer[idx + offset]->id, idx + offset, lane_id);
            }
#endif
        }
    }
}

#ifdef RUN_WITH_TESTS
#define CHECK_FOR_ERROR() { cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());}
#else
//#define CHECK_FOR_ERROR() { cudaDeviceSynchronize(); gpuErrchk(cudaPeekAtLastError());}
#define CHECK_FOR_ERROR() { gpuErrchk(cudaPeekAtLastError()); }
#endif



__global__
void MultiPreScan(BucketMemory *container, TrafficObject_id **objectsToInsert, size_t *n, unsigned int *temp_value) {
    for(size_t idx = GetGlobalIdx(); idx < *n; idx += GetGlobalDim()) {

        int insert_offset = atomicAdd(temp_value + objectsToInsert[idx]->lane, 1);
        auto &bucket = container->buckets[objectsToInsert[idx]->lane];
        bucket.buffer[bucket.size + insert_offset] = objectsToInsert[idx];
#ifdef RUN_WITH_TESTS
        if (objectsToInsert[idx]->id == CAR_TO_ANALYZE) {
            printf("Car(%lu) from ReinsertBuffer(#%lu) to Bucket(%lu)\n", objectsToInsert[idx]->id, idx, objectsToInsert[idx]->lane);
        }
#endif
    }
}


__global__
void SetTempSizeKernel(BucketMemory *container, unsigned int *temp_value) {
    size_t lane_id = GetGlobalIdx();

    assert(GetGlobalDim() >= container->bucket_count);

    if (lane_id < container->bucket_count) {
        container->buckets[lane_id].size += temp_value[lane_id];
#ifdef RUN_WITH_TESTS
        if (lane_id == BUCKET_TO_ANALYZE)
            printf("New Temporary Bucket(%lu) size: %lu\n", (size_t) BUCKET_TO_ANALYZE,
                   container->buckets[lane_id].size);
#endif
    }
}

#define PRE_SUM_BLOCK_SIZE 512

class PreSumBuffer {
public:
    std::vector<size_t *> temporary_pre_sum_buffers;
    std::vector<size_t> temporary_pre_sum_buffer_sizes;
    size_t *preSum;

    unsigned int *multiScanTmp;
    TrafficObject_id **reinsert_buffer;
    size_t reinsert_buffer_size;
    size_t multiScanTmpBytes;
    PreSumBuffer(Scenario_id &scenario) {
        multiScanTmpBytes = 2 * scenario.lanes.size() * sizeof(unsigned int);
        gpuErrchk(cudaMalloc((void**) &multiScanTmp, multiScanTmpBytes * sizeof(unsigned int )));

        reinsert_buffer_size = scenario.lanes.size();
        gpuErrchk(cudaMalloc((void**) &reinsert_buffer, reinsert_buffer_size * sizeof(TrafficObject_id*)));

        size_t buffer_size = BucketMemory::getBufferSize(scenario, 4.);
        assert(IsPowerOfTwo(PRE_SUM_BLOCK_SIZE));

        gpuErrchk(cudaMalloc((void**) &preSum, buffer_size * sizeof(size_t)));
        temporary_pre_sum_buffers.push_back(preSum);
        temporary_pre_sum_buffer_sizes.push_back(buffer_size);


        size_t preSumTempSize = buffer_size;
        size_t *preSumTemp;
        for(size_t i=PRE_SUM_BLOCK_SIZE; i < buffer_size; i *= PRE_SUM_BLOCK_SIZE) {
            preSumTempSize = preSumTempSize / PRE_SUM_BLOCK_SIZE + 1;
            gpuErrchk(cudaMalloc((void **) &preSumTemp, preSumTempSize * sizeof(size_t)));
            temporary_pre_sum_buffers.push_back(preSumTemp);
            temporary_pre_sum_buffer_sizes.push_back(preSumTempSize);
        }
    }

    ~PreSumBuffer() {
        for(size_t * buffer : temporary_pre_sum_buffers) {
            gpuErrchk(cudaFree(buffer));
        }

        gpuErrchk(cudaFree(multiScanTmp));
        gpuErrchk(cudaFree(reinsert_buffer));

    }
};

void collect_changed(Scenario_id &scenario, BucketMemory *container, PreSumBuffer &preSumBuffer) {
    size_t buffer_size = BucketMemory::getBufferSize(scenario, 4.);
    assert(IsPowerOfTwo(PRE_SUM_BLOCK_SIZE));

    size_t number_of_lanes = scenario.lanes.size();
    size_t number_of_cars = scenario.cars.size();

    std::vector<size_t> preSumHost(buffer_size);

    BlockWisePreScan<<<buffer_size / PRE_SUM_BLOCK_SIZE + 1, PRE_SUM_BLOCK_SIZE / 2, PRE_SUM_BLOCK_SIZE * sizeof(size_t)>>>(container, preSumBuffer.preSum, PRE_SUM_BLOCK_SIZE);
    CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
    gpuErrchk(cudaMemcpy(preSumHost.data(), preSumBuffer.temporary_pre_sum_buffers[0], preSumBuffer.temporary_pre_sum_buffer_sizes[0] * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_FOR_ERROR();
#endif

    size_t *previousPreSumTemp, *preSumTemp;
    size_t previousPreSumTempSize, preSumTempSize;
    for(size_t i=1; i < preSumBuffer.temporary_pre_sum_buffers.size(); i++) {
        preSumTemp = preSumBuffer.temporary_pre_sum_buffers[i];
        preSumTempSize = preSumBuffer.temporary_pre_sum_buffer_sizes[i];
        previousPreSumTemp = preSumBuffer.temporary_pre_sum_buffers[i - 1];
        previousPreSumTempSize = preSumBuffer.temporary_pre_sum_buffer_sizes[i - 1];

        MergeBlockWisePreScan<<<preSumTempSize / PRE_SUM_BLOCK_SIZE + 1, PRE_SUM_BLOCK_SIZE / 2, PRE_SUM_BLOCK_SIZE * sizeof(size_t )>>>(preSumTemp, preSumTempSize, previousPreSumTemp, previousPreSumTempSize, PRE_SUM_BLOCK_SIZE, PRE_SUM_BLOCK_SIZE);
        CHECK_FOR_ERROR();

#ifdef RUN_WITH_TESTS
        gpuErrchk(cudaMemcpy(preSumHost.data(), preSumTemp, preSumTempSize * sizeof(size_t), cudaMemcpyDeviceToHost));
        CHECK_FOR_ERROR();
#endif
    }

    for(size_t i=preSumBuffer.temporary_pre_sum_buffer_sizes.size() - 1; i > 0; i--) {
        preSumTemp = preSumBuffer.temporary_pre_sum_buffers[i];
        preSumTempSize = preSumBuffer.temporary_pre_sum_buffer_sizes[i];
        previousPreSumTemp = preSumBuffer.temporary_pre_sum_buffers[i - 1];
        previousPreSumTempSize = preSumBuffer.temporary_pre_sum_buffer_sizes[i - 1];

#ifdef RUN_WITH_TESTS
        gpuErrchk(cudaMemcpy(preSumHost.data(), preSumTemp, preSumTempSize * sizeof(size_t), cudaMemcpyDeviceToHost));
        CHECK_FOR_ERROR();
#endif

        MergeBlockWisePreScanStep2<<<preSumTempSize, PRE_SUM_BLOCK_SIZE, PRE_SUM_BLOCK_SIZE * sizeof(size_t )>>>(previousPreSumTemp, preSumTemp, PRE_SUM_BLOCK_SIZE, previousPreSumTempSize, previousPreSumTempSize);
        CHECK_FOR_ERROR();
    }


#ifdef RUN_WITH_TESTS
    gpuErrchk(cudaMemcpy(preSumHost.data(), preSumBuffer.temporary_pre_sum_buffers[0], preSumBuffer.temporary_pre_sum_buffer_sizes[0] * sizeof(size_t), cudaMemcpyDeviceToHost));
    CHECK_FOR_ERROR();
#endif

    MoveToReinsertBuffer<<<buffer_size / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>(container, preSumBuffer.preSum, buffer_size, preSumBuffer.reinsert_buffer, preSumBuffer.reinsert_buffer_size);
    CHECK_FOR_ERROR()

    gpuErrchk(cudaMemsetAsync(preSumBuffer.multiScanTmp, 0, preSumBuffer.multiScanTmpBytes));
    MultiPreScan<<<256, 256>>>(container, preSumBuffer.reinsert_buffer, preSumBuffer.preSum + buffer_size - 1, preSumBuffer.multiScanTmp);
    CHECK_FOR_ERROR();

    SetTempSizeKernel<<<number_of_lanes, 1>>>(container, preSumBuffer.multiScanTmp);
    CHECK_FOR_ERROR();

/*
    InsertChanged<<<512, 16, 32 * sizeof(size_t )>>>(container, 32, preSum, buffer_size, reinsert_buffer, reinsert_buffer_size);
    CHECK_FOR_ERROR()
*/

    dim3 blocks(MIN(2048, number_of_lanes), 1);
    dim3 threads(512, 1); // TODO longest lane...

    TrafficObject_id::Cmp cmp;
    cudaSortKernel<<<blocks, threads>>>(container, cmp);
    CHECK_FOR_ERROR();

    FixSizeKernel<<<256, 20>>>(container, true);
    CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
    /*TrafficObject_id *bla[buffer_size];
    BucketMemory mem;
    gpuErrchk(cudaMemcpy(&mem, container, sizeof(BucketMemory), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(bla, mem.main_buffer, buffer_size * sizeof(TrafficObject_id*), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(preSumHost.data(), preSum, buffer_size * sizeof(size_t), cudaMemcpyDeviceToHost));*/
#endif

}

__global__ void prescank(size_t *tmp) {
    PreScan(tmp, GetThreadIdx(), GetBlockDim() * 2, 1);
}


void TestAlgo::advance(size_t steps) {

    Scenario_id scenario = *getIDScenario();
    PreSumBuffer preSumBuffer(scenario);
    size_t number_of_lanes = scenario.lanes.size();
    size_t number_of_cars = scenario.cars.size();

    CudaScenario_id *device_cuda_scenario = CudaScenario_id::fromScenarioData_device(scenario);

    std::shared_ptr<BucketMemory> bucket_memory = BucketMemory::fromScenario(scenario, device_cuda_scenario);

    TrafficObject_id **dev_left_neighbors, **dev_own_neighbors, **dev_right_neighbors;
    gpuErrchk(cudaMalloc((void**) &dev_left_neighbors, 2 * number_of_cars * sizeof(TrafficObject_id*)));
    gpuErrchk(cudaMalloc((void**) &dev_own_neighbors, 2 * number_of_cars * sizeof(TrafficObject_id*)));
    gpuErrchk(cudaMalloc((void**) &dev_right_neighbors, 2 * number_of_cars * sizeof(TrafficObject_id*)));

    Car_id::AdvanceData *device_changes;
    gpuErrchk(cudaMalloc((void**) &device_changes, getIDScenario()->cars.size() * sizeof(Car_id::AdvanceData)));

#ifdef DEBUG_MSGS
    printf("Starting to advance scenario...\n\n");
#endif
#ifdef RUN_WITH_TESTS
    printf("Car(%lu) on Lane(%lu)\n", (size_t) CAR_TO_ANALYZE, scenario.cars[CAR_TO_ANALYZE].lane);
#endif
    for(int i = 0; i < steps; i++) {
#ifdef DEBUG_MSGS
        printf("Step: %d\n", i);
#endif

        collect_changed(scenario, bucket_memory.get(), preSumBuffer);

#ifdef RUN_WITH_TESTS
        testBucketsForInvalidLaneKernel<<<number_of_lanes, 1024>>>(bucket_memory.get(), device_cuda_scenario);
        CHECK_FOR_ERROR();
#endif

        find_nearest<<<number_of_cars * 3 / SUGGESTED_THREADS + 1, SUGGESTED_THREADS>>>
            (device_cuda_scenario, bucket_memory.get(), dev_left_neighbors, dev_own_neighbors, dev_right_neighbors);
        CHECK_FOR_ERROR();

#ifdef RUN_WITH_TESTS
        test_neighborsKernel<<<512, 512>>>(device_cuda_scenario, dev_left_neighbors, dev_own_neighbors, dev_right_neighbors);
        CHECK_FOR_ERROR();
#endif

        kernel_get_changes<<<512, 512>>>(device_changes, device_cuda_scenario, dev_right_neighbors, dev_own_neighbors, dev_left_neighbors);
        CHECK_FOR_ERROR();

#ifdef RUN_WITH_TESTS
        testChangesKernel<<<512, 512>>>(device_cuda_scenario, device_changes);
        CHECK_FOR_ERROR();
#endif

        applyChangesKernel<<<512, 512>>>(device_changes, device_cuda_scenario);
        CHECK_FOR_ERROR();

        updateSignalsKernel<<<512, 512>>>(device_cuda_scenario);
        CHECK_FOR_ERROR();

    }

    device_cuda_scenario->retriveData(getIDScenario());

};