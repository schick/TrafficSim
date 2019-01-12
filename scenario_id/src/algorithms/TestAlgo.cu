//
// Created by oke on 09.01.19.
//

#include <curand_mtgp32_kernel.h>
#include <model/Lane.h>
#include "algorithms/TestAlgo.h"
#include "cuda/cuda_utils.h"


struct BucketData {
    size_t size;
    size_t buffer_size;
    TrafficObject_id **buffer;
};

class BucketMemory {
public:
    size_t bucket_count;
    BucketData *buckets;
    TrafficObject_id **main_buffer;
    size_t main_buffer_size;


    CUDA_HOSTDEV BucketMemory() : bucket_count(0), buckets(nullptr), main_buffer(nullptr), main_buffer_size(0) {

    }

    CUDA_HOSTDEV static size_t getBufferSize(CudaScenario_id *scenario, float bucket_memory_factor) {
        size_t total_buffer_size = 0;
        for (Lane_id &l : scenario->getLaneIterator()) {
            total_buffer_size += ceil(bucket_memory_factor * scenario->getRoad(l.road)->length / 5.);
        }
        return total_buffer_size;
    }

    CUDA_HOSTDEV BucketMemory(CudaScenario_id *scenario, BucketData *_buckets, TrafficObject_id **_main_buffer, float bucket_memory_factor) {
        bucket_count = scenario->getNumLanes();
        buckets = _buckets; // new BucketData[scenario->getNumLanes()];
        assert(buckets != nullptr);

        size_t total_buffer_size = 0;
        size_t i = 0;
        for (Lane_id &l : scenario->getLaneIterator()) {
            buckets[i].size = 0;
            buckets[i].buffer_size = ceil(bucket_memory_factor * scenario->getRoad(l.road)->length / 5.);
            total_buffer_size += buckets[i].buffer_size;
            i++;
        }

        main_buffer = _main_buffer; // new TrafficObject_id*[total_buffer_size];
        assert(main_buffer != nullptr);
        this->main_buffer_size = total_buffer_size;

        size_t offset = 0;
        for(i = 0; i < scenario->getNumLanes(); i++) {
            buckets[i].buffer = main_buffer + offset;
            offset += buckets[i].buffer_size;
        }
        printf("Allocated: %.2fMB\n", (float) (sizeof(size_t) * scenario->getNumLanes() * 2 + sizeof(TrafficObject_id**) * scenario->getNumLanes() +
                                               sizeof(TrafficObject_id*) * total_buffer_size) / 1024. / 1024.);
    }

    CUDA_HOSTDEV ~BucketMemory() {
        //if (buckets != nullptr) free(buckets);
        //if (main_buffer != nullptr) free(main_buffer);
    }



};


template<> template<>
BucketContainer<TrafficObject_id*, nullptr> BucketContainer<TrafficObject_id*, nullptr>::construct<CudaScenario_id>(CudaScenario_id &scenario) {
    BucketContainer<TrafficObject_id *, nullptr> bucketContainer(scenario.getNumLanes(), 15);
    for(auto &c : scenario.getCarIterator()) {
        auto &bucket = bucketContainer[c.lane];
        bucket.resize(bucket.getSize() + 1);
        bucket[bucket.getSize() - 1] = &c;
    }
    assert(bucketContainer.numElements() == scenario.getNumCars());
    return bucketContainer;

}

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


__global__ void allocateKernel(CudaScenario_id *device_scenario, BucketContainer<TrafficObject_id *, nullptr> *device_bucketContainer) {
    size_t lane_idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (lane_idx == 0) *device_bucketContainer = BucketContainer<TrafficObject_id *, nullptr>(device_scenario->getNumLanes(), 0);
}

__global__ void constructKernel(CudaScenario_id *device_scenario, BucketContainer<TrafficObject_id *, nullptr> *device_bucketContainer) {
    size_t lane_idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (lane_idx >= device_bucketContainer->getNumBuckets()) return;

    auto &bucket = (*device_bucketContainer)[lane_idx];

    for(auto &c : device_scenario->getCarIterator()) {
        if(lane_idx == c.lane) {
            bucket.resize(50);
            continue;
            bucket.resize(bucket.getSize() + 1);
            bucket[bucket.getSize() - 1] = &c;
        }
    }
    //printf("%lu, %lu ---- %lu/%lu: %lu\n", (size_t )threadIdx.x, (size_t )blockIdx.x, lane_idx, device_bucketContainer->getNumBuckets(), bucket.getSize());

}
__global__ void checkKernel(CudaScenario_id *device_scenario, BucketContainer<TrafficObject_id *, nullptr> *device_bucketContainer) {
    assert(device_bucketContainer->numElements() == device_scenario->getNumCars());
}

template<> template<>
void BucketContainer<TrafficObject_id*, nullptr>::construct_device<CudaScenario_id>(CudaScenario_id *device_scenario, BucketContainer<TrafficObject_id *, nullptr> *device_bucketContainer, size_t num_lanes) {

    unsigned int BLOCK_NUM = ceil((float) num_lanes / THREAD_NUM);
    dim3 threads(THREAD_NUM, 1);
    dim3 blocks(BLOCK_NUM, 1);
    allocateKernel<<<1, 1>>>(device_scenario, device_bucketContainer);
    constructKernel<<<blocks, threads>>>(device_scenario, device_bucketContainer);
#ifdef DEBUG_MSGS
    checkKernel<<<1, 1>>>(device_scenario, device_bucketContainer);
#endif
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

template <typename Cmp>
__global__ void cudaSortKernel(BucketMemory *container, Cmp cmp) {
    size_t first_bucket_idx = blockIdx.x + gridDim.x * blockIdx.y;
    size_t i = threadIdx.x + blockDim.x * threadIdx.y;
    size_t block_count = gridDim.x * gridDim.y;
    size_t thread_count = blockDim.x * blockDim.y;
    size_t buckets_per_thread = ceilf((float) container->bucket_count / block_count);

    for(size_t bucket_idx = first_bucket_idx; bucket_idx < container->bucket_count; bucket_idx += block_count) {
#if DEBUG_MSGS
        // if (first_bucket_idx == 0 && i == 0) printf("bucket_idx: %lu - block count: %lu\n", bucket_idx, block_count);
#endif
        if (container->buckets[bucket_idx].size > 1) {
            TrafficObject_id **device_values = container->buckets[bucket_idx].buffer;
            size_t n = container->buckets[bucket_idx].size;
            // printf("bucket %lu with %lu items.\n", bucket_idx, n);
            int j, k;
            /* Major step */
            for (k = 2; k <= n; k <<= 1) {
                /* Minor step */
                for (j = k >> 1; j > 0; j = j >> 1) {
                    bitonic_sort_step(device_values, j, k, n, cmp);
                    __syncthreads();
                }
            }

            unsigned long power = pow(2., floor(log((double) n) / log(2.)));
            for (unsigned long k = power; k > 0; k >>= 1) {
                bitonic_sort_merge(device_values, k, n, cmp);
                __syncthreads();
            }
        }
    }
}
__global__ void checkSortKernel(BucketMemory *container) {
    TrafficObject_id::Cmp cmp;
    size_t bucket_id = GetGlobalIdx();

    if (bucket_id >= container->bucket_count) return;
    if(bucket_id == 0) printf("Testing Sorting of Container...\n");
    if (container->buckets[bucket_id].size == 0) return;
    //printf("Lane %d\n", i);
    for(int j = 0; j < container->buckets[bucket_id].size - 1; j++) {
        //printf("%lu,%lu: %f <= %f (%d, %d)\n", buckets[i][j]->id, buckets[i][j + 1]->id,  buckets[i][j]->x, buckets[i][j + 1]->x,
        //        cmp(buckets[i][j], buckets[i][j + 1]), buckets[i][j]->x <= buckets[i][j + 1]->x);
        assert(cmp(container->buckets[bucket_id].buffer[j], container->buckets[bucket_id].buffer[j + 1]) && container->buckets[bucket_id].buffer[j]->x <= container->buckets[bucket_id].buffer[j + 1]->x);
    }
}


template <typename ObjectType, ObjectType Z, typename Cmp>
__global__ void sort_cu(BucketContainer<ObjectType, Z> *container, Cmp cmp)
{
    __shared__ int res;
    res = 0;
    size_t fid = -1;
    int bucket_id = blockIdx.y;
    auto &bucket = (*container)[bucket_id];

    int i = blockIdx.x;
    int j = threadIdx.x;

    if(bucket_id == fid && j == 0 && i == 0) printf("index(%d-%d) size: %lu\n", i, j, bucket.getSize());
    if(i >= bucket.getSize())
        return;


    __shared__ ObjectType array[1024];

    if(j < bucket.getSize())
        array[j] = bucket[j];

    __syncthreads();

    if(j < bucket.getSize()) {
        if (bucket_id == fid) printf("index(%d-%d) %p/%p\n", i, j, array[i], array[j]);
        //if (bucket_id == fid) printf("%p: %p\n", array + j, array[j]);
        if (cmp(array[j], array[i]) || (i > j && !cmp(array[i], array[j]))) {
            // if ((array[i] > array[j]) || (i > j && array[i] == array[j])) {
            atomicAdd((unsigned int *) &res, 1);
            if(bucket_id == fid) printf("index(%d-%d): %f/%f - %d\n", i, j, array[i]->x, array[j]->x , res);
        }
    }

    __syncthreads();

    if(bucket_id == fid && j == 0) printf("index(%d): %d\n", i, res);
    if(j < bucket.getSize())
        bucket[res] = array[i];
}

template <typename ObjectType, ObjectType Z>
template<typename Cmp>
void BucketContainer<ObjectType, Z>::sort_device_bucket(BucketContainer<ObjectType, Z> *container, Cmp cmp, size_t num_buckets) {

    /*dim3 blocks(1024, num_buckets);
    dim3 threads(50, 1);
    sort_cu<<<blocks, threads>>>(container, cmp);
    gpuErrchk( cudaPeekAtLastError() );*/
    dim3 blocks(MIN(2048, num_buckets), 1);    /* Number of blocks   */
    dim3 threads(20, 1);  /* Number of threads  */
//    cudaSortKernel<<<blocks, threads>>>(container, cmp);

}




__global__ void find_nearest(CudaScenario_id *scenario, BucketMemory *container, TrafficObject_id **nearest_left,
        TrafficObject_id **nearest_own, TrafficObject_id **nearest_right) {
    AlgorithmWrapper wrapper(*scenario);
#ifdef DEBUG_MSGS
    size_t print_for_id = -1;
#endif
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
                if (car.id == print_for_id)
                    printf("Find(%lu): %lu/%.2f, Current(%lu) %lu/%.2f, Index: %lu/%lu/%lu \n", car.id, car.lane, car.x,
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
            if (car.id == print_for_id)
                printf("Find(%lu): %lu/%.2f, Current(%lu) %lu/%.2f, Index: %lu/%lu/%lu \n", car.id, car.lane, car.x,
                       lane_objects[search_idx]->id, lane_objects[search_idx]->lane, lane_objects[search_idx]->x, from,
                       search_idx, to);
#endif

            assert(search_idx < n && (*lane_objects[search_idx] < car || search_idx == 0));

            if (search_idx == 0 && *lane_objects[search_idx] >= car) {
                nearest_back = nullptr;
                while (search_idx < n && *lane_objects[search_idx] == car) search_idx++;
                if (search_idx < n)
                    nearest_font = lane_objects[search_idx];
            } else {
                if (search_idx < n)
                    nearest_back = lane_objects[search_idx];
                search_idx++;
                while (search_idx < n && *lane_objects[search_idx] == car) search_idx++;
                if (search_idx < n)
                    nearest_font = lane_objects[search_idx];
            }
        }
    }

#ifdef DEBUG_MSGS
    if(car.id == print_for_id) {
        printf("Found(%lu): %lu %lu\n", car.id,
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
    printf("test_right_lane_neighbors\n");
    AlgorithmWrapper algorithmWrapper(*scenario);
    size_t car_id = GetGlobalIdx();
    if (car_id >= scenario->getNumLanes()) return;

    Road_id::NeighboringLanes lanes = algorithmWrapper.getNeighboringLanes(*scenario->getLane(scenario->getCar(car_id)->lane));
    if(lanes.right == (size_t) -1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
        assert(neighbors[car_id] == nullptr);
        return;
    }
    Lane_id::NeighboringObjects neig = algorithmWrapper.getNeighboringObjects(*scenario->getCar(car_id), *scenario->getLane(lanes.right));

    if (neig.back == -1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
    } else {
        assert(neighbors[scenario->getNumCars() +car_id ] != nullptr && neighbors[scenario->getNumCars() +car_id]->id == neig.back);
    }

    if (neig.front == -1) {
        if(neighbors[car_id] != nullptr) printf("%lu\n", neighbors[car_id]->id);

        assert(neighbors[car_id] == nullptr);
    } else {
        assert(neighbors[car_id] != nullptr && neighbors[car_id]->id == neig.front);
    }
}

__device__ void test_left_lane_neighbors(TrafficObject_id **neighbors, CudaScenario_id *scenario) {
    printf("test_left_lane_neighbors\n");
    AlgorithmWrapper algorithmWrapper(*scenario);
    size_t car_id = GetGlobalIdx();
    if (car_id >= scenario->getNumLanes()) return;

    Road_id::NeighboringLanes lanes = algorithmWrapper.getNeighboringLanes(*scenario->getLane(scenario->getCar(car_id)->lane));
    if(lanes.left == (size_t) -1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
        assert(neighbors[car_id] == nullptr);
        return;
    }
    Lane_id::NeighboringObjects neig = algorithmWrapper.getNeighboringObjects(*scenario->getCar(car_id), *scenario->getLane(lanes.left));

    if (neig.back == -1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
    } else {
        assert(neighbors[scenario->getNumCars() +car_id ] != nullptr && neighbors[scenario->getNumCars() +car_id]->id == neig.back);
    }

    if (neig.front == -1) {
        assert(neighbors[car_id] == nullptr);
    } else {
        assert(neighbors[car_id] != nullptr && neighbors[car_id]->id == neig.front);
    }
}

__device__ void test_own_lane_neighbors(TrafficObject_id **neighbors, CudaScenario_id *scenario) {
    printf("test_own_lane_neighbors\n");
    AlgorithmWrapper algorithmWrapper(*scenario);
    size_t car_id = GetGlobalIdx();
    if (car_id >= scenario->getNumLanes()) return;

    Lane_id::NeighboringObjects neig = algorithmWrapper.getNeighboringObjects(*scenario->getCar(car_id), *scenario->getLane(scenario->getCar(car_id)->lane));

    if (neig.back == -1) {
        assert(neighbors[scenario->getNumCars() + car_id] == nullptr);
    } else {
        assert(neighbors[scenario->getNumCars() +car_id ] != nullptr && neighbors[scenario->getNumCars() +car_id]->id == neig.back);
    }

    if (neig.front == -1) {
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

__global__
void bucketMemoryInitializeKernel(BucketMemory *bucketmem,  BucketData *buckets, TrafficObject_id **main_buffer, CudaScenario_id *cuda_device_scenario, float bucket_memory_factor) {
    new(bucketmem)BucketMemory(cuda_device_scenario, buckets, main_buffer, bucket_memory_factor);
}

__global__
void bucketMemoryLoadKernel(BucketMemory *bucketmem, CudaScenario_id *cuda_device_scenario, float bucket_memory_factor) {
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

__global__ void kernel_get_changes(Car_id::AdvanceData *change, CudaScenario_id * scenario_data,
                                   TrafficObject_id **right_lane_neighbors, TrafficObject_id **own_lane_neighbors, TrafficObject_id **left_lane_neighbors) {
    size_t car_idx = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    AlgorithmWrapper algorithm_wrapper(*scenario_data);
    if (car_idx < scenario_data->getNumCars()) {
        Lane_id::NeighboringObjectsRef rightNeighbors(right_lane_neighbors[car_idx], right_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        Lane_id::NeighboringObjectsRef ownNeighbors(own_lane_neighbors[car_idx], own_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        Lane_id::NeighboringObjectsRef leftNeighbors(left_lane_neighbors[car_idx], left_lane_neighbors[car_idx + scenario_data->getNumCars()]);
        change[car_idx] = algorithm_wrapper.nextStep(*scenario_data->getCar(car_idx), leftNeighbors, ownNeighbors, rightNeighbors);
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

__global__ void testBucketsForInvalidLaneKernel(BucketMemory *container) {
    size_t bucket_id = GetGlobalIdx();
    if (bucket_id >= container->bucket_count) return;
    if(bucket_id == 0) printf("testing container buckets\n");
    if (container->buckets[bucket_id].size == 0) return;
    //printf("Lane %d\n", i);
    for(int j = 0; j < container->buckets[bucket_id].size - 1; j++) {
        assert(container->buckets[bucket_id].buffer[j]->lane == bucket_id);
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

    __syncthreads();
    extern __shared__ size_t temp[];// allocated on invocation

    size_t traffic_object_id = GetThreadIdx();
    size_t buffer_offset = GetBlockIdx() * n;


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
    __shared__ unsigned long int count;
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

__global__ void MergeBlockWisePreScan(size_t *out, size_t *in, size_t n, size_t buffer_size, size_t skip_count) {
    assert(IsPowerOfTwo(n));

    size_t idx = GetThreadIdx();
    size_t offset = GetBlockIdx() * n * skip_count;

    extern __shared__ size_t temp[];
    if ((idx + 1) * skip_count - 1 < buffer_size)
        temp[2 * idx] = in[(2 * idx + 1) * skip_count - 1 + offset];
    else
        temp[2 * idx] = 0;

    if ((idx + 1) * skip_count - 1 < buffer_size)
        temp[2 * idx + 1] = in[(2 * idx + 2) * skip_count - 1 + offset];
    else
        temp[2 * idx + 1] = 0;

    __syncthreads();

    PreScan(temp, idx, n);

    __syncthreads();

    out[2 * idx + n * GetBlockIdx()] = temp[2 * idx];
    out[2 * idx + 1 + n * GetBlockIdx()] = temp[2 * idx + 1];
}

__global__ void MergeBlockWisePreScanStep2(size_t *out, size_t *in, size_t n, size_t buffer_size, size_t out_size) {
    size_t idx = GetThreadIdx();
    size_t offset = GetBlockIdx() * n;

    if (GetBlockIdx() > 0 && idx + offset < out_size) {
        out[idx + offset] += in[GetBlockIdx() - 1];
    }

}

__global__ void GetIndicesKernel(BucketMemory *container, size_t *prefixSum, size_t n,
        TrafficObject_id **reinsert_buffer, size_t buffer_size) {
    size_t idx = GetGlobalIdx();
    if (idx >= n) return;
    // if(idx == 256) printf("%lu - %lu\n" , prefixSum[255], prefixSum[256]);
    if ((idx == 0 && prefixSum[0] > 0) || (idx != 0 && prefixSum[idx] != prefixSum[idx - 1])){
        // printf("%lu\n", container->main_buffer_size);
        size_t insert_id = idx == 0 ? 0 : prefixSum[idx - 1];
        reinsert_buffer[insert_id] = container->main_buffer[idx];
        container->main_buffer[idx] = nullptr;
        /*printf("Buffer(%lu)_chng: %lu -> %lu, car(%lu)\n", idx, idx == 0 ? 0 : prefixSum[idx - 1], prefixSum[idx],
                container->main_buffer[idx] == nullptr ? -1 : container->main_buffer[idx]->id);*/

        // container->buckets[container->main_buffer[idx - 1]->lane][prefixSum[idx - 1]]
    }

    __syncthreads();
    //printf("%lu,%lu - %lu,%lu,%lu|%lu,%lu,%lu\n", prefixSum[100800], prefixSum[100801],
    //        prefixSum[98301], prefixSum[98302], prefixSum[98303], prefixSum[98304], prefixSum[98305], prefixSum[98306]);
}

__global__ void InsertChanged(BucketMemory *container, size_t bufferSize,
        size_t *prefixSum, size_t prefixSumLength, TrafficObject_id **reinsertBuffer, size_t reinsertBufferLength) {

    assert(IsPowerOfTwo(bufferSize));

    size_t idx = GetThreadIdx();
    size_t lane_id = GetBlockIdx();

    extern __shared__ size_t insert_into_lane[];


    if (2 * idx + 1 >= bufferSize) return;
    if (prefixSum[prefixSumLength - 1] == 0) return;
    if(2 * idx < prefixSum[prefixSumLength - 1])
        insert_into_lane[2 * idx] = (size_t) (reinsertBuffer[2 * idx]->lane == lane_id);
    else
        insert_into_lane[2 * idx] = 0;


    if(2 * idx + 1 < prefixSum[prefixSumLength - 1])
        insert_into_lane[2 * idx + 1] = (size_t) (reinsertBuffer[2 * idx + 1]->lane == lane_id);
    else
        insert_into_lane[2 * idx + 1] = 0;

    PreScan(insert_into_lane, idx, bufferSize);

    __syncthreads();

    // printf("Insert %lu new Cars.\n", insert_into_lane[bufferSize - 1]);

    auto &bucket = container->buckets[lane_id];
    if (idx >= prefixSum[prefixSumLength - 1]) return;
    if ((idx == 0 && insert_into_lane[0] > 0) || (idx != 0 && insert_into_lane[idx] != insert_into_lane[idx - 1])) {
        size_t insert_id = idx == 0 ? 0 : insert_into_lane[idx - 1];
        assert(bucket.buffer[bucket.size + insert_id] == nullptr);
        //printf("lane: %lu, bucket idx: %lu\n", lane_id, insert_id + bucket.size);
        bucket.buffer[bucket.size + insert_id] = reinsertBuffer[idx];
    }

    __syncthreads();

    if(idx == 0) bucket.buffer_size += insert_into_lane[bufferSize - 1];

}

__global__ void FixSizeKernel(BucketMemory *container) {

    size_t idx = GetThreadIdx();
    size_t lane_id = GetBlockIdx();

    auto &bucket = container->buckets[lane_id];

    __shared__ size_t new_size;
    if (idx >= bucket.buffer_size) return;
    if (idx == 0) {
        if (bucket.buffer[0] == nullptr)
            new_size = 0;
    } else {
        if (bucket.buffer[idx] == nullptr && bucket.buffer[idx - 1] != nullptr)
            new_size = idx;
    }

    __syncthreads();

    if (idx == 0) {
       // printf("%lu: %lu\n", lane_id, new_size);
        bucket.size = new_size;
    }
}

#define CHECK_FOR_ERROR() cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

void collect_changed(CudaScenario_id *scenario, BucketMemory *container) {

#ifdef RUN_WITH_TESTS
    collect_changedKernel<<<1024,1024>>>(container);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
#endif

    size_t buffer_size = BucketMemory::getBufferSize(scenario, 4.);
    size_t block_size = 512;
    assert(IsPowerOfTwo(block_size));

    std::vector<size_t> preSumHost(buffer_size);

    size_t *preSum;
    gpuErrchk(cudaMalloc((void**) &preSum, buffer_size * sizeof(size_t)));

    BlockWisePreScan<<<ceil((float) buffer_size / block_size), block_size / 2, block_size * sizeof(size_t)>>>(container, preSum, block_size);
    CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
#endif

    std::vector<size_t *> reduce_arrays;
    reduce_arrays.push_back(preSum);
    std::vector<size_t> reduce_sizes;
    reduce_sizes.push_back(buffer_size);

    size_t initial_temp_size = ceil((float) buffer_size / block_size);
    size_t temp_size = initial_temp_size;
    size_t block_size_i;
    for(block_size_i=block_size; block_size_i < buffer_size; block_size_i *= block_size) {

        size_t *temp;
        gpuErrchk(cudaMalloc((void**) &temp, temp_size * sizeof(size_t) ));

        MergeBlockWisePreScan<<<ceil((float) temp_size / block_size), block_size / 2., block_size * sizeof(size_t )>>>(temp, reduce_arrays.back(), block_size, reduce_sizes.back(), block_size);

        // printf("%f\n",ceil((float) temp_size / block_size) );
        CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk(cudaMemcpy(preSumHost.data(), temp, temp_size * sizeof(size_t), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
#endif
        reduce_arrays.push_back(temp);
        reduce_sizes.push_back(temp_size);

        temp_size = ceil((float)temp_size / block_size);
    }
    while(reduce_arrays.size() != 1) {
        size_t *temp = reduce_arrays.back();
        reduce_arrays.pop_back();
        temp_size = reduce_sizes.back();
        reduce_sizes.pop_back();

#ifdef RUN_WITH_TESTS
        gpuErrchk(cudaMemcpy(preSumHost.data(), temp, temp_size * sizeof(size_t), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
#endif

        MergeBlockWisePreScanStep2<<<temp_size, block_size, block_size * sizeof(size_t )>>>(reduce_arrays.back(), temp, block_size, reduce_sizes.back(), reduce_sizes.back());
        CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
#endif

        gpuErrchk( cudaFree(temp) );
    }


#ifdef RUN_WITH_TESTS
    gpuErrchk(cudaMemcpy(preSumHost.data(), reduce_arrays.back(), reduce_sizes.back() * sizeof(size_t), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
#endif

    TrafficObject_id **reinsert_buffer;
    gpuErrchk(cudaMalloc((void**) &reinsert_buffer, 400 * sizeof(TrafficObject_id*)));

    GetIndicesKernel<<<buffer_size / 256 + 1, 256>>>(container, preSum, buffer_size, reinsert_buffer, 400.);
    CHECK_FOR_ERROR()

    InsertChanged<<<scenario->getNumLanes(), 256, 512 * sizeof(size_t )>>>(container, 512, preSum, buffer_size, reinsert_buffer, 400.);
    CHECK_FOR_ERROR()

    dim3 blocks(MIN(2048, scenario->getNumLanes()), 1);    /* Number of blocks   */
    dim3 threads(20, 1);  /* Number of threads  */

    TrafficObject_id::Cmp cmp;
    cudaSortKernel<<<blocks, threads>>>(container, cmp);
    CHECK_FOR_ERROR();

    FixSizeKernel<<<scenario->getNumLanes(), 1024>>>(container);
    CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    TrafficObject_id *bla[buffer_size];
    BucketMemory mem;
    gpuErrchk(cudaMemcpy(&mem, container, sizeof(BucketMemory), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(bla, mem.main_buffer, buffer_size * sizeof(TrafficObject_id*), cudaMemcpyDeviceToHost));


    gpuErrchk(cudaMemcpy(preSumHost.data(), preSum, buffer_size * sizeof(size_t), cudaMemcpyDeviceToHost));
    // cudaDeviceSynchronize();

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    printf("collect_changed---\n");
#endif

    gpuErrchk(cudaFree(preSum));
}

__global__ void prescank(size_t *tmp) {
    PreScan(tmp, GetThreadIdx(), GetBlockDim() * 2, 1);
}

void TestAlgo::advance(size_t steps) {
/*
    constexpr size_t arr_size = 16;
    size_t ceil_arr_size = pow(2, ceil(log(arr_size)/log(2)));
    size_t my_array[arr_size];
    size_t prefix_sum[arr_size];
    size_t i;
    for (i = 0; i < arr_size; i++) {
        my_array[i] = rand() > std::numeric_limits<int>::max() / 2 ? 1 : 0;
    }

    size_t *dev_my_array;
    gpuErrchk(cudaMalloc((void**) &dev_my_array, ceil_arr_size * sizeof(size_t)));
    size_t *dev_my_array_sorted;
    gpuErrchk(cudaMalloc((void**) &dev_my_array_sorted, ceil_arr_size * sizeof(size_t)));


    gpuErrchk(cudaMemcpy(dev_my_array, my_array, ceil_arr_size * sizeof(size_t), cudaMemcpyHostToDevice));

    //dev_mem_bitonic_sort(dev_my_array, arr_size);
    prescank<<<1, ceil_arr_size/2>>>(dev_my_array);

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk(cudaMemcpy(prefix_sum, dev_my_array, arr_size * sizeof(size_t), cudaMemcpyDeviceToHost));

    size_t offset = 0;
    for(int i = 0; i < arr_size; i++) {
        offset += my_array[i];
        printf("%lu, %lu, %lu\n", prefix_sum[i], offset, my_array[i]);
        //assert(prefix_sum[i] == offset);
    }
    printf("\n");
    exit(0);*/
    /*
    constexpr size_t arr_size = 1024;
    int my_array[arr_size];
    int i;
    for (i = 0; i < arr_size; i++) {
        my_array[i] = rand();
    }

    int *dev_my_array;
    gpuErrchk(cudaMalloc((void**) &dev_my_array, arr_size * sizeof(int)));
    int *dev_my_array_sorted;
    gpuErrchk(cudaMalloc((void**) &dev_my_array_sorted, arr_size * sizeof(int)));

    gpuErrchk(cudaMemcpy(dev_my_array, my_array, arr_size * sizeof(int), cudaMemcpyHostToDevice));

    //dev_mem_bitonic_sort(dev_my_array, arr_size);
    sort_cu<<<arr_size, arr_size>>>(dev_my_array, dev_my_array_sorted);
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk(cudaMemcpy(my_array, dev_my_array_sorted, arr_size * sizeof(int), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();
    //std::cout << my_array[0] << std::endl;
    for (i = 1; i < arr_size; i++) {
        //std::cout << my_array[i] << std::endl;
        if(my_array[i - 1] > my_array[i]) {
            std::cout << "ERROR." << std::endl;
            exit(0);
        }
    }*/
    Scenario_id *__scenario = getIDScenario();
    CudaScenario_id scenario = CudaScenario_id::fromScenarioData(*__scenario);

    CudaScenario_id *device_cuda_scenario = CudaScenario_id::fromScenarioData_device(*__scenario);

    TrafficObject_id **main_buffer;
    gpuErrchk(cudaMalloc((void**) &main_buffer, BucketMemory::getBufferSize(&scenario, 4) * sizeof(TrafficObject_id*)));

    BucketData *buckets;
    gpuErrchk(cudaMalloc((void**) &buckets, scenario.getNumLanes() * sizeof(BucketData)));

    // allocate bucket class
    BucketMemory *bucket_memory;
    gpuErrchk(cudaMalloc((void**) &bucket_memory, sizeof(BucketMemory)));

    // initialize bucket class
    bucketMemoryInitializeKernel<<<1, 1>>>(bucket_memory, buckets, main_buffer, device_cuda_scenario, 4.);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    // load data into memory
    bucketMemoryLoadKernel<<<1024, 1024>>>(bucket_memory, device_cuda_scenario, 4.);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    TrafficObject_id **dev_left_neighbors, **dev_own_neighbors, **dev_right_neighbors;
    gpuErrchk(cudaMalloc((void**) &dev_left_neighbors, 2 * scenario.getNumCars() * sizeof(TrafficObject_id*)));
    gpuErrchk(cudaMalloc((void**) &dev_own_neighbors, 2 * scenario.getNumCars() * sizeof(TrafficObject_id*)));
    gpuErrchk(cudaMalloc((void**) &dev_right_neighbors, 2 * scenario.getNumCars() * sizeof(TrafficObject_id*)));


    TrafficObject_id::Cmp cmp;

    dim3 blocks(MIN(2048, scenario.getNumLanes()), 1);    /* Number of blocks   */
    dim3 threads(20, 1);  /* Number of threads  */

    Car_id::AdvanceData *device_changes;
    gpuErrchk(cudaMalloc((void**) &device_changes, getIDScenario()->cars.size() * sizeof(Car_id::AdvanceData)));

/*
    dim3 blocks(ceil(sqrt(scenario.getNumCars())), ceil(sqrt(scenario.getNumCars())));
    dim3 threads(1, 3);
*/
    // test_neighbors(device_bucketContainer, device_cuda_scenario, &scenario);
#ifdef DEBUG_MSGS
    printf("start steps\n");
#endif

    for(int i = 0; i < steps; i++) {
#ifdef DEBUG_MSGS
        printf("Step: %d\n", i);
#endif
        /*BucketContainer<TrafficObject_id *, nullptr>::sort_device_bucket(device_bucketContainer, cmp, scenario.getNumLanes());
        gpuErrchk( cudaPeekAtLastError() );*/

        cudaSortKernel<<<blocks, threads>>>(bucket_memory, cmp);
        CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        checkSortKernel<<<scenario.getNumLanes(), 1>>>(bucket_memory);

        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
#endif

        find_nearest<<<blocks, threads>>>(device_cuda_scenario, bucket_memory, dev_left_neighbors, dev_own_neighbors, dev_right_neighbors);
        CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        test_neighborsKernel<<<1, 1>>>(device_cuda_scenario, dev_left_neighbors, dev_own_neighbors, dev_right_neighbors);

        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
#endif

        kernel_get_changes<<<512, 512>>>(device_changes, device_cuda_scenario, dev_right_neighbors, dev_own_neighbors, dev_left_neighbors);
        CHECK_FOR_ERROR()

#ifdef RUN_WITH_TESTS

        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        testChangesKernel<<<512, 512>>>(device_cuda_scenario, device_changes);

        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

#endif
        applyChangesKernel<<<512, 512>>>(device_changes, device_cuda_scenario);
        CHECK_FOR_ERROR()

        collect_changed(&scenario, bucket_memory);

        // prescan<<<bucket_memory->bucket_count, ceil_arr_size/2, ceil_arr_size * sizeof(size_t)>>>(dev_my_array_sorted, dev_my_array, arr_size, ceil_arr_size);

#ifdef RUN_WITH_TESTS

        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        testBucketsForInvalidLaneKernel<<<512, 512>>>(bucket_memory);

        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

#endif
        updateSignalsKernel<<<512, 512>>>(device_cuda_scenario);
        gpuErrchk( cudaPeekAtLastError() );
    }

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    device_cuda_scenario->retriveData(getIDScenario());

/*    gpuErrchk(cudaFree(device_changes));
    gpuErrchk( cudaPeekAtLastError() );*/

};