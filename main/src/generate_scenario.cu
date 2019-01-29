#include "util/SimpleArgumentParser.h"
#include <curand_kernel.h>
#include <cuda_utils/cuda_utils.h>
#include <string>
#include "generate_scenario.h"



__global__ void init_stuff(curandState *state) {
    size_t idx = GetGlobalIdx();
    curand_init(1337, idx, 0, &state[idx]);
}


__global__ void select_junctions(curandState *state, unsigned int *junction_idxs, float probabilty, junction *junctions, unsigned int *junciton_idx) {
    size_t junction_dim = GetBlockDim();
    size_t junction_x = GetThreadIdx();
    size_t junction_y = GetBlockIdx();

    if(curand_uniform(&state[junction_x + junction_y * junction_dim]) < probabilty) {
        unsigned int idx = atomicAdd(junciton_idx, 1);
        junctions[idx] = {idx, junction_x, junction_y};
        size_t si = 0;
        for(auto &s : junctions[idx].signals) { s.time = 0; s.dir = si; si++; }
        junction_idxs[junction_x + junction_y * junction_dim] = idx;
    } else {
        junction_idxs[junction_x + junction_y * junction_dim] = (size_t ) -1;
    }
}

__device__ size_t calcDirectionOfRoad(junction &from, junction &to) {
    // linkshändisches koordinatensystemv
    if (from.y < to.y) {
        return 2;
    } else if (from.y > to.y) {
        return 0;
    } else if (from.x < to.x) {
        return 1;
    } else if (from.x > to.x) {
        return 3;
    }
}


__global__ void select_roads(curandState *state, unsigned int *use_junction, road *roads, float probabilty, unsigned int *roads_idx,
                             unsigned int *total_length, junction *junctions) {
    size_t junction_dim = GetBlockDim();
    size_t junction_x = GetThreadIdx();
    size_t junction_y = GetBlockIdx();
    bool right = junction_y % 2 == 0;
    junction_y /= 2;

    road road;
    road.from_x = junction_x;
    road.from_y = junction_y;
    road.to_x = junction_x;
    road.to_y = junction_y;
    road.lanes = 0;
    road.length = 0;

    if(use_junction[road.from_x + road.from_y * junction_dim] != -1 && curand_uniform(&state[junction_x + GetBlockIdx() * GetBlockDim()]) < probabilty) {
        road.from_id = use_junction[road.from_x + road.from_y * junction_dim];
        if (right) {
            road.to_x++;
            road.length++;
            while(use_junction[road.to_x + road.to_y * junction_dim] == -1 && road.to_x < junction_dim) {
                road.to_x++;
                road.length++;
            }
            if (road.to_x == junction_dim) {
                return;
            }
            assert(use_junction[road.from_x + road.from_y * junction_dim] != -1);
            assert(use_junction[road.to_x + road.to_y * junction_dim] != -1);
            float p = curand_uniform(&state[junction_x + GetBlockIdx() * GetBlockDim()]);
            road.lanes = p < 0.2 ? 1 : (p < 0.8 ? 2 : 3);
        } else {
            road.to_y++;
            road.length++;
            while(use_junction[road.to_x + road.to_y * junction_dim]  == -1&& road.to_y < junction_dim) {
                road.to_y++;
                road.length++;
            }
            if (road.to_y == junction_dim) {
                return;
            }
            assert(use_junction[road.from_x + road.from_y * junction_dim] != -1);
            assert(use_junction[road.to_x + road.to_y * junction_dim] != -1);
            float p = curand_uniform(&state[junction_x + GetBlockIdx() * GetBlockDim()]);
            road.lanes = (p < 0.2) ? 1 : ((p < 0.8) ? 2 : 3);
        }
        if(road.lanes != 0) {
            junction &to = junctions[use_junction[road.to_x + road.to_y * junction_dim]];
            junction &from = junctions[use_junction[road.from_x + road.from_y * junction_dim]];
            from.signals[calcDirectionOfRoad(from, to)].time = curand_uniform(&state[junction_x + GetBlockIdx() * GetBlockDim()]) * 7 + 5;
            to.signals[calcDirectionOfRoad(to, from)].time = curand_uniform(&state[junction_x + GetBlockIdx() * GetBlockDim()]) * 7 + 5;
            road.to_id = use_junction[road.to_x + road.to_y * junction_dim];
            road.limit = curand_uniform(&state[junction_x + GetBlockIdx() * GetBlockDim()]) * 60 + 30;
            assert(road.from_x < junction_dim && road.from_y < junction_dim && road.to_x < junction_dim && road.to_y < junction_dim);
            assert(use_junction[road.from_x + road.from_y * junction_dim] != -1);
            assert(use_junction[road.to_x + road.to_y * junction_dim] != -1);
            roads[atomicAdd(roads_idx, 1)] = road;
            atomicAdd(total_length,  road.length);
        }
    }
}

__global__ void select_cars(curandState *state, car *cars, road *roads, unsigned int road_count) {
    size_t car_idx = GetGlobalIdx();

    size_t road_idx = curand_uniform(&state[GetGlobalIdx()]) * road_count;
    size_t lane = curand_uniform(&state[GetGlobalIdx()]) * roads[road_idx].lanes;

    car c;
    c.target_deceleration = curand_uniform(&state[GetGlobalIdx()]) * 2 + 2;
    c.max_acceleration = curand_uniform(&state[GetGlobalIdx()]) * 2 + 1;
    c.target_headway = curand_uniform(&state[GetGlobalIdx()]) * 2 + 1;
    c.politeness = curand_uniform(&state[GetGlobalIdx()]) * 0.8 + 0.1;
    c.target_velocity = curand_uniform(&state[GetGlobalIdx()]) * 70 + 70;
    c.min_distance = curand_uniform(&state[GetGlobalIdx()]) * 2 + 2;
    c.from = roads[road_idx].from_id;
    c.to = roads[road_idx].to_id;
    for(auto &r : c.route) r = curand_uniform(&state[GetGlobalIdx()]) * 6;
    c.id = car_idx;
    c.lane = lane;
    c.distance = curand_uniform(&state[GetGlobalIdx()]) * 100 * roads[road_idx].length;
    cars[car_idx] = c;

}

int get_random(int argc, char *argv[], std::vector<car> &host_cars, std::vector<junction> &host_junctions, std::vector<road> &host_roads) {
    SimpleArgumentParser parser;
    parser.add_kw_argument("grid_size", "512");
    parser.add_kw_argument("junction_prob", "0.5");
    parser.add_kw_argument("road_prob", "0.7");

    parser.load(argc, argv);

    size_t gridSize = std::stoul(parser["grid_size"]);
    float junction_prob = std::stof(parser["junction_prob"]);
    float road_prob = std::stof(parser["road_prob"]);
    int each = 5;

    size_t maxThreads = gridSize;
    size_t maxBlocks = gridSize * 2 * each;


    curandState *d_state;
    cudaMalloc(&d_state, maxThreads * maxBlocks * sizeof(curandState));;
    init_stuff<<<maxBlocks, maxThreads>>>(d_state);
    CHECK_FOR_ERROR();

    unsigned int *junction_counter;
    gpuErrchk(cudaMalloc(&junction_counter, sizeof(unsigned int )));
    gpuErrchk(cudaMemset(junction_counter, 0, sizeof(unsigned int )));

    unsigned int *junction_idxs;
    gpuErrchk(cudaMalloc(&junction_idxs, gridSize * gridSize * sizeof(unsigned int)));
    junction *junctions;
    gpuErrchk(cudaMalloc(&junctions, gridSize * gridSize * sizeof(junction)));

    select_junctions<<<gridSize, gridSize>>>(d_state, junction_idxs, junction_prob, junctions, junction_counter);
    CHECK_FOR_ERROR();

    road *roads;
    gpuErrchk(cudaMalloc(&roads, gridSize * gridSize * sizeof(road)))
    unsigned int *counter;
    gpuErrchk(cudaMalloc(&counter, sizeof(unsigned int )));
    gpuErrchk(cudaMemset(counter, 0, sizeof(unsigned int )));

    unsigned int *total_length;
    gpuErrchk(cudaMalloc(&total_length, sizeof(unsigned int )));
    gpuErrchk(cudaMemset(total_length, 0, sizeof(unsigned int )));

    select_roads<<<2 * gridSize, gridSize>>>(d_state, junction_idxs, roads, road_prob, counter, total_length, junctions);
    CHECK_FOR_ERROR();



    unsigned int host_junctions_counter;
    gpuErrchk(cudaMemcpy(&host_junctions_counter, junction_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    host_junctions.resize(host_junctions_counter);
    gpuErrchk(cudaMemcpy(host_junctions.data(), junctions, host_junctions_counter * sizeof(junction), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(&host_junctions_counter, counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    host_roads.resize(host_junctions_counter);
    gpuErrchk(cudaMemcpy(host_roads.data(), roads, host_junctions_counter * sizeof(road), cudaMemcpyDeviceToHost));

    CHECK_FOR_ERROR();

    int car_count = host_junctions_counter * each;
    car_count = ((int)sqrt(car_count) + 1) * ((int)sqrt(car_count) + 1);


    /*cudaMalloc(&d_state, car_count * sizeof(curandState));;
    init_stuff<<<sqrt(car_count), sqrt(car_count)>>>(d_state);
    CHECK_FOR_ERROR();*/

    car *cars;
    gpuErrchk(cudaMalloc((void**)&cars, car_count * sizeof(car)))
    select_cars<<<sqrt(car_count), sqrt(car_count)>>>(d_state, cars, roads, host_junctions_counter);


    host_cars.resize(car_count);
    gpuErrchk(cudaMemcpy(host_cars.data(), cars, car_count * sizeof(car), cudaMemcpyDeviceToHost));


    cudaDeviceSynchronize();
}