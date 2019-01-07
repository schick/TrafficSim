#include <RedTrafficLight_id.h>
#include <cudacontainer.h>
#include <Road_id.h>
#include "Junction_id.h"
#include "Car_id.h"
#include "cuda/cuda_utils.h"

template<typename T>
__device__ void cuda_swap(T &t1, T &t2) { T t = t1; t1 = t2; t2 = t; }

template<typename T>
__global__ void bitonic_sort_step(T *dev_values, int j, int k, int n) {
    unsigned int i, ixj; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
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
__global__ void bitonic_sort_merge(T* values, int k, int n) {
    unsigned int i; /* Sorting partners: i and ixj */
    i = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
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
            bitonic_sort_step<<<blocks, threads>>>(device_values, j, k, n);
            gpuErrchk( cudaPeekAtLastError() );
        }
    }

    unsigned long power = pow(2, floor(log(n)/log(2)));
    for (unsigned long k = power; k > 0; k >>= 1) {
        bitonic_sort_merge<<<blocks, threads>>>(device_values, k, n);
        gpuErrchk( cudaPeekAtLastError() );
    }
}


__global__ void lower_bound(const TrafficObject_id *find_values, size_t *nearest_font, size_t *nearest_back, size_t find_n, const TrafficObject_id* value, size_t n) {
    size_t i = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    size_t iinv = (size_t )-1;
    if (i >= find_n)
        return;
    const TrafficObject_id &find = find_values[i];
    if (find.lane == (size_t )-1) {
        if(find_values[i].id == iinv) {
            printf("Find(%lu): %lu/%.2f, No lane...\n", find.id, find.lane, find.x);
        }
        nearest_back[i] = (size_t) -1;
        nearest_font[i] = (size_t) -1;
        return;
    }
    size_t search_idx = n / 2;
    size_t from = 0;
    size_t to = n;
    while(true) {
        if(find_values[i].id == iinv) {
           printf("Find(%lu): %lu/%.2f, Current(%lu) %lu/%.2f, Index: %lu/%lu/%lu \n", find.id, find.lane, find.x, value[search_idx].id, value[search_idx].lane, value[search_idx].x, from, search_idx, to);
        }
        if (value[search_idx] < find) {
            if (value[search_idx + 1] >= find) {
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
    if(find_values[i].id == iinv) {
        printf("Find(%lu): %lu/%.2f, Current(%lu) %lu/%.2f, Index: %lu/%lu/%lu \n", find.id, find.lane, find.x, value[search_idx].id, value[search_idx].lane, value[search_idx].x, from, search_idx, to);
    }
    assert(search_idx < n && (value[search_idx] < find || search_idx == 0));

    if (search_idx == 0 && value[search_idx] >= find) {
        if(find_values[i].id == iinv)printf("%lu =< %lu\n", find.lane, value[search_idx].lane);
        nearest_back[i] = (size_t ) -1;
        if(find_values[i].id == iinv)printf("%lu\n", search_idx);
        while (search_idx < n && value[search_idx] == find) search_idx++;
        if(find_values[i].id == iinv)printf("%lu\n", search_idx);
        nearest_font[i] = search_idx;
    } else {

        nearest_back[i] = search_idx;
        search_idx++;
        if(find_values[i].id == iinv)printf("%lu\n", search_idx);
        while (value[search_idx] == find && search_idx < n) search_idx++;
        nearest_font[i] = search_idx;
    }
    if (nearest_back[i] == n) nearest_back[i] = (size_t ) -1;
    if (nearest_font[i] == n) nearest_font[i] = (size_t ) -1;
    if (nearest_back[i] != (size_t ) -1 && value[nearest_back[i]].lane != find_values[i].lane)
        nearest_back[i] = (size_t ) -1;
    if (nearest_font[i] != (size_t ) -1 && value[nearest_font[i]].lane != find_values[i].lane)
        nearest_font[i] = (size_t ) -1;

    if(find_values[i].id == iinv) {
        printf("Found(%lu): %lu %lu\n", find.id,
               (nearest_back[i] != (size_t) -1 ? value[nearest_back[i]].id: (size_t) -1),
                nearest_font[i] != (size_t) -1 ? value[nearest_font[i]].id: (size_t) -1);
    }
}



__global__ void put_on_lane_device_kernel(TrafficObject_id *out, const TrafficObject_id *device_objects, size_t n,
        size_t *other_lane_id, int8_t offset) {
    size_t i = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    out[i] = device_objects[i];
    for(uint8_t idx = 0; idx < offset; idx++) {
        if (i < n && other_lane_id[out[i].lane] != (size_t) -1 && out[i].lane != (size_t )-1) {
            // printf("New Lane of Car %lu is %lu\n", out[i].id, other_lane_id[out[i].lane]);
            out[i].lane = (int) other_lane_id[out[i].lane];
        } else {
            out[i].lane = (size_t) -1;
            break;
        }
    }
}

TrafficObject_id MAX_TOBJ(std::numeric_limits<size_t>::min(), 0, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max());



__host__ SortedUniqueObjectContainer::SortedUniqueObjectContainer(size_t container_count, size_t lane_count) :
        device_container_objects(nullptr), container_size(container_count), lane_count(lane_count),
        host_container_objects(container_count) {
    container_size_2_ceil = pow(2, ceil( log(container_size) / log(2)));
}

__host__ SortedUniqueObjectContainer::~SortedUniqueObjectContainer() {
    deallocate_device_memory();
}

__host__ void SortedUniqueObjectContainer::allocate_device_memory() {
    if (isAllocated) return;
    gpuErrchk(cudaMalloc((void**) &device_container_objects, container_size_2_ceil * sizeof(TrafficObject_id)));
    std::vector<TrafficObject_id> maximums((container_size_2_ceil - container_size), MAX_TOBJ);
    gpuErrchk(cudaMemcpy(device_container_objects + container_size, maximums.data(),
                         (container_size_2_ceil - container_size) * sizeof(TrafficObject_id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &device_right_lane_id, lane_count * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void**) &device_left_lane_id, lane_count * sizeof(size_t)));

    gpuErrchk(cudaMalloc((void**) &device_front_idx,container_size_2_ceil  * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void**) &device_back_idx, container_size_2_ceil * sizeof(size_t)));

    isAllocated = true;
}

__host__ void SortedUniqueObjectContainer::deallocate_device_memory() {
    if (!isAllocated) return;
    transfer_objects_to_host();
    assert(isValidOnHost);
    gpuErrchk(cudaFree(device_container_objects));
    gpuErrchk(cudaFree(device_front_idx));
    gpuErrchk(cudaFree(device_back_idx));
    gpuErrchk(cudaFree(device_left_lane_id));
    gpuErrchk(cudaFree(device_right_lane_id));
    isAllocated = false;
    isValidOnDevice = false;
}

__host__ void SortedUniqueObjectContainer::get_nearest_objects_hd(const TrafficObject_id* device_objects,
        std::vector<size_t > &nearest_front, std::vector<size_t > &nearest_back, int8_t lane_offset) {
    assert(nearest_back.size() == nearest_front.size());

    size_t *front_idx, *back_idx;
    gpuErrchk(cudaMalloc((void**) &front_idx, nearest_front.size() * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void**) &back_idx, nearest_front.size() * sizeof(size_t)));

    get_nearest_objects_device(device_objects, front_idx, back_idx, nearest_front.size(), lane_offset);

    gpuErrchk(cudaMemcpy(nearest_back.data(), back_idx, nearest_back.size() * sizeof(size_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(nearest_front.data(), front_idx, nearest_back.size() * sizeof(size_t), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(front_idx));
    gpuErrchk(cudaFree(back_idx));

}
__host__ void SortedUniqueObjectContainer::get_nearest_objects_device(const TrafficObject_id* device_objects,
        size_t *device_nearest_front, size_t *device_nearest_back, size_t n, int8_t lane_offset) {
    assert(isLaneInformationSet);

    unsigned long block_num = (unsigned int) ceil(n / (float) THREADS);
    dim3 blocks(block_num, 1);
    dim3 threads(THREADS, 1);

    transfer_objects_to_device();

    TrafficObject_id *device_objects_copy = nullptr;
    if (lane_offset != 0) {
        gpuErrchk(cudaMalloc((void**) &device_objects_copy, n * sizeof(TrafficObject_id)));
        put_on_lane_device_kernel<<<blocks, threads>>> (device_objects_copy, device_objects, n,
                (lane_offset > 0) ? device_right_lane_id : device_left_lane_id, abs(lane_offset));
        gpuErrchk( cudaPeekAtLastError() );
        device_objects = device_objects_copy;
    }

    sort_data();

    lower_bound<<<blocks, threads>>>(device_objects, device_nearest_front, device_nearest_back, n,
            device_container_objects, container_size);
    gpuErrchk( cudaPeekAtLastError() );

    if (device_objects_copy != nullptr) {
        gpuErrchk(cudaFree(device_objects_copy));
    }
}


TrafficObject_id* SortedUniqueObjectContainer::createDeviceObjects(const std::vector<TrafficObject_id> &objects) {
    TrafficObject_id *dev;

    gpuErrchk(cudaMalloc((void**) &dev, objects.size() * sizeof(TrafficObject_id)));
    gpuErrchk(cudaMemcpy(dev, objects.data(),  objects.size() * sizeof(TrafficObject_id), cudaMemcpyHostToDevice));
    return dev;
}

void SortedUniqueObjectContainer::deleteDeviceObjects(TrafficObject_id* objects) {
    gpuErrchk(cudaFree(objects));
}

void SortedUniqueObjectContainer::setDeviceObjects(TrafficObject_id* device_objects, std::vector<TrafficObject_id> new_objects){
    gpuErrchk(cudaMemcpy(device_objects, new_objects.data(),  new_objects.size() * sizeof(TrafficObject_id), cudaMemcpyHostToDevice));
}

__host__ void SortedUniqueObjectContainer::get_nearest_objects_host(
        const std::vector<TrafficObject_id> &host_objects, std::vector<size_t> &nearest_front, std::vector<size_t> &nearest_back,
        int8_t lane_offset) {

    assert(nearest_back.size() == nearest_front.size() && nearest_back.size() == host_objects.size());

    size_t n = nearest_back.size();
    size_t *front_idx, *back_idx;
    gpuErrchk(cudaMalloc((void**) &front_idx, n * sizeof(size_t)));
    gpuErrchk(cudaMalloc((void**) &back_idx, n * sizeof(size_t)));

    TrafficObject_id *device_objects;
    gpuErrchk(cudaMalloc((void**) &device_objects, n * sizeof(TrafficObject_id)));
    gpuErrchk(cudaMemcpy(device_objects, host_objects.data(), n * sizeof(TrafficObject_id), cudaMemcpyHostToDevice));

    get_nearest_objects_device(device_objects, front_idx, back_idx, n, lane_offset);

    gpuErrchk(cudaMemcpy(nearest_back.data(), back_idx, n * sizeof(size_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(nearest_front.data(), front_idx, n * sizeof(size_t), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(front_idx));
    gpuErrchk(cudaFree(back_idx));
}

void SortedUniqueObjectContainer::sort_data() {
    if (isSorted) return;
    transfer_objects_to_device();
    dev_mem_bitonic_sort(device_container_objects, container_size_2_ceil);
    isSorted = true;
    isValidOnHost = false;
}

__host__ const std::vector<TrafficObject_id> &SortedUniqueObjectContainer::get_objects() {
    transfer_objects_to_host();
    return host_container_objects;
}

__host__ std::vector<TrafficObject_id> &SortedUniqueObjectContainer::get_rw_objects() {
    transfer_objects_to_host();
    isValidOnDevice = false;
    isSorted = false;
    return host_container_objects;
}

__host__ std::vector<TrafficObject_id> &SortedUniqueObjectContainer::get_w_objects() {
    isValidOnDevice = false;
    isSorted = false;
    isValidOnHost = true;
    return host_container_objects;
}

__host__ void SortedUniqueObjectContainer::transfer_objects_to_host() {
    if (isValidOnHost) return;
    assert(isAllocated);
    gpuErrchk(cudaMemcpy(host_container_objects.data(), device_container_objects, container_size * sizeof(TrafficObject_id), cudaMemcpyDeviceToHost));
    isValidOnHost = true;
}

__host__ void SortedUniqueObjectContainer::transfer_objects_to_device() {
    allocate_device_memory();
    if (isValidOnDevice) return;
    gpuErrchk(cudaMemcpy(device_container_objects, host_container_objects.data(), container_size * sizeof(TrafficObject_id), cudaMemcpyHostToDevice));
    isValidOnDevice = true;
}

void SortedUniqueObjectContainer::setLaneInformation(const std::vector<size_t> &leftLaneId, const std::vector<size_t> &rightLaneId) {
    assert(leftLaneId.size() == lane_count && rightLaneId.size() == lane_count);
    allocate_device_memory();
    gpuErrchk(cudaMemcpy(device_left_lane_id, leftLaneId.data(), lane_count * sizeof(size_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_right_lane_id, rightLaneId.data(), lane_count * sizeof(size_t), cudaMemcpyHostToDevice));
    isLaneInformationSet = true;
}

template <typename T>
void host_mem_bitonic_sort(T *values, size_t n, T max)
{
    /*
        T *dev_values;
        size_t size = n * sizeof(T);
        unsigned long power = pow(2, ceil(log(n)/log(2)));

        gpuErrchk(cudaMalloc((void**) &dev_values, power));
        T* max_values = (T*) malloc(sizeof(T) * (power - n));
        gpuErrchk(cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_values + n, max_values, size, cudaMemcpyHostToDevice));
        free(max_values);
        */

    size_t *hfront, *hback;
    hfront = (size_t*)malloc(n * sizeof(size_t));
    hback = (size_t*)malloc(n * sizeof(size_t));

    T *dev_values;
    unsigned long power = pow(2, ceil(log(n)/log(2)));
    T* max_values = (T*) malloc(sizeof(T) * (power - n));
    for (int i= 0; i < (power - n); i++) max_values[i] = max;
    assert(max_values != nullptr);

    size_t data_size = n * sizeof(T);
    size_t max_size = (power - n) * sizeof(T);
    size_t mem_size = power * sizeof(T);

    gpuErrchk(cudaMalloc((void**) &dev_values, mem_size));
    gpuErrchk(cudaMemcpy(dev_values, values, data_size, cudaMemcpyHostToDevice));



    gpuErrchk(cudaMemcpy(dev_values + n, max_values, max_size, cudaMemcpyHostToDevice));


    dev_mem_bitonic_sort(dev_values, power);

    size_t block_num = (unsigned int) ceil(n / (float) THREADS);
    size_t block_num2 = 1;
    printf("%d Threads on %lux%lu Blocks\n", THREADS, block_num, block_num2);
    if (block_num > 65535) {
        block_num2 = 65535;
        block_num = (size_t) ceil((float) block_num / (float) block_num2);
    }
    dim3 blocks(block_num, block_num2);    /* Number of blocks   */
    dim3 threads(THREADS, 1);  /* Number of threads  */

    size_t *front, *back;
    gpuErrchk(cudaMalloc((void**) &front, data_size));
    gpuErrchk(cudaMalloc((void**) &back, data_size));

    lower_bound<<<blocks, threads>>>(dev_values, front, back, n, dev_values, n);
    gpuErrchk( cudaPeekAtLastError() );


    gpuErrchk(cudaMemcpy(values, dev_values, data_size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hfront, front, n * sizeof(size_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(hback, back, n * sizeof(size_t), cudaMemcpyDeviceToHost));

    for(int i=0; i < n; i++) {
        printf("%f, %f, %f\n", hback[i] == n ? NAN : values[hback[i]].k1, values[i].k1, hfront[i] == n ? NAN : values[hfront[i]].k1);
    }

    cudaFree(front);
    cudaFree(back);
    cudaFree(dev_values);
}
/**
get_nearest_objects_
cars[i].nextStep
advanceStep
*/