#ifndef cudacontainer
#define cudacontainer

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include <iostream>
#include <chrono>
#include <thread>

#include "TrafficObject_id.h"

class SortedUniqueObjectContainer {
public:
    SortedUniqueObjectContainer(size_t container_count, size_t lane_count);
    ~SortedUniqueObjectContainer();

    void allocate_device_memory();
    void deallocate_device_memory();

    void transfer_objects_to_host();
    void transfer_objects_to_device();

    std::vector<TrafficObject_id> &get_rw_objects();
    const std::vector<TrafficObject_id> &get_objects();
    std::vector<TrafficObject_id> &get_w_objects();

    void sort_data();

    void setLaneInformation(const std::vector<size_t> &leftLaneId, const std::vector<size_t> &rightLaneId);

    void get_nearest_objects_host(const std::vector<TrafficObject_id> &host_objects,
                                  std::vector<size_t> &nearest_front, std::vector<size_t> &nearest_back, int8_t lane_offset=0);
    void get_nearest_objects_device(const TrafficObject_id* device_objects,
                                    size_t *device_nearest_front, size_t *device_nearest_back, size_t n, int8_t lane_offset=0);

    void get_nearest_objects_hd(const TrafficObject_id* device_objects,
            std::vector<size_t > &nearest_front, std::vector<size_t > &nearest_back, int8_t lane_offset=0);


    TrafficObject_id *createDeviceObjects(const std::vector<TrafficObject_id> &objects);
    void deleteDeviceObjects(TrafficObject_id* objects);
    void setDeviceObjects(TrafficObject_id* device_objects, std::vector<TrafficObject_id> new_objects);

private:

    bool isAllocated = false;
    bool isValidOnHost = true;
    bool isValidOnDevice = false;
    bool isLaneInformationSet = false;
    bool isSorted = false;

    size_t lane_count;
    size_t container_size;
    size_t container_size_2_ceil;

    // car memory
    TrafficObject_id *device_container_objects;
    std::vector<TrafficObject_id> host_container_objects;
    size_t *device_front_idx;
    size_t *device_back_idx;
    size_t *device_left_front_idx;
    size_t *device_left_back_idx;
    size_t *device_right_front_idx;
    size_t *device_right_back_idx;

    size_t *device_left_lane_id;
    size_t *device_right_lane_id;
};


#endif
