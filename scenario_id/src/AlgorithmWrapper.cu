//
// Created by oke on 22.12.18.
//



#include "AlgorithmWrapper.h"
#include "cuda/cuda_utils.h"

#define C_MIN(a, b) (a < b ? a : b)
#define C_MAX(a, b) (a < b ? b : a)

__global__ void  get_nearest_objects(CudaScenario_id* scenario, size_t *nearest_objects) {
    size_t i = threadIdx.x + blockDim.x * blockIdx.x + blockDim.y * blockIdx.y;
    if (i < scenario->getNumCars()) {
        // scenario->getCar(i)->nextStep(scenario);
    }
    // cu_getNeighboringObjects(scenario, &scenario->cars[i], &scenario->lanes[scenario->cars[i].lane]);
}

CudaScenario_id *CudaScenario_id::fromScenarioData_device(ScenarioData_id &scenario) {
    CudaScenario_id cudaScenarioId;
    gpuErrchk(cudaMalloc((void**) &cudaScenarioId.junctions, scenario.junctions.size() * sizeof(Junction_id)));
    gpuErrchk(cudaMemcpy(cudaScenarioId.junctions, scenario.junctions.data(), scenario.junctions.size() * sizeof(Junction_id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &cudaScenarioId.signals, scenario.signals.size() * sizeof(Junction_id::Signal)));
    gpuErrchk(cudaMemcpy(cudaScenarioId.signals, scenario.signals.data(), scenario.signals.size() * sizeof(Junction_id::Signal), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &cudaScenarioId.traffic_lights, scenario.traffic_lights.size() * sizeof(RedTrafficLight_id)));
    gpuErrchk(cudaMemcpy(cudaScenarioId.traffic_lights, scenario.traffic_lights.data(), scenario.traffic_lights.size() * sizeof(RedTrafficLight_id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &cudaScenarioId.roads, scenario.roads.size() * sizeof(Road_id)));
    gpuErrchk(cudaMemcpy(cudaScenarioId.roads, scenario.roads.data(), scenario.roads.size() * sizeof(Road_id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &cudaScenarioId.lanes, scenario.lanes.size() * sizeof(Lane_id)));
    gpuErrchk(cudaMemcpy(cudaScenarioId.lanes, scenario.lanes.data(), scenario.lanes.size() * sizeof(Lane_id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &cudaScenarioId.cars, scenario.cars.size() * sizeof(Car_id)));
    gpuErrchk(cudaMemcpy(cudaScenarioId.cars, scenario.cars.data(), scenario.cars.size() * sizeof(Car_id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void**) &cudaScenarioId.turns, scenario.turns.size() * sizeof(Car_id::TurnDirection)));
    gpuErrchk(cudaMemcpy(cudaScenarioId.turns, scenario.turns.data(), scenario.turns.size() * sizeof(Car_id::TurnDirection), cudaMemcpyHostToDevice));

    cudaScenarioId.num_cars = scenario.cars.size();
    cudaScenarioId.num_junctions = scenario.junctions.size();
    cudaScenarioId.num_lanes = scenario.lanes.size();
    cudaScenarioId.num_lights = scenario.traffic_lights.size();
    cudaScenarioId.num_roads = scenario.roads.size();
    cudaScenarioId.num_signals = scenario.signals.size();
    cudaScenarioId.num_turns = scenario.turns.size();

    CudaScenario_id *device_cuda_scenario;
    gpuErrchk(cudaMalloc((void**) &device_cuda_scenario, sizeof(CudaScenario_id)));
    gpuErrchk(cudaMemcpy(device_cuda_scenario, &cudaScenarioId, sizeof(CudaScenario_id), cudaMemcpyHostToDevice));
    return device_cuda_scenario;
}

void CudaScenario_id::freeDeviceCudaScenario(CudaScenario_id *device_cuda_scenario){
    CudaScenario_id &cudaScenarioId = *device_cuda_scenario;

    gpuErrchk(cudaFree(cudaScenarioId.junctions));
    gpuErrchk(cudaFree(cudaScenarioId.signals));
    gpuErrchk(cudaFree(cudaScenarioId.traffic_lights));
    gpuErrchk(cudaFree(cudaScenarioId.roads));
    gpuErrchk(cudaFree(cudaScenarioId.lanes));
    gpuErrchk(cudaFree(cudaScenarioId.cars));
    gpuErrchk(cudaFree(cudaScenarioId.turns));
    gpuErrchk(cudaFree(cudaScenarioId.turns));

    gpuErrchk(cudaFree(device_cuda_scenario));


}


void CudaScenario_id::retriveCars(Car_id *host_cars) {
    CudaScenario_id host_scenario;
    gpuErrchk(cudaMemcpy(&host_scenario, this, sizeof(CudaScenario_id), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(host_cars, host_scenario.cars, host_scenario.num_cars * sizeof(Car_id), cudaMemcpyDeviceToHost));
}

//
// Created by oke on 22.12.18.
//


#include "AlgorithmWrapper.h"
#include "math.h"


CUDA_HOSTDEV Lane_id::NeighboringObjects AlgorithmWrapper::getNeighboringObjects(const TrafficObject_id &object, Lane_id& lane) {
    // TODO: assert is sorted

    Lane_id::NeighboringObjects result;

    const TrafficObject_id *closest_gt = nullptr;
    const TrafficObject_id *closest_lt = nullptr;

    TrafficObject_id::Cmp cmp;
    for(size_t i=0; i < s.getNumCars(); i++) {
        TrafficObject_id *car = s.getCar(i);
        if(car->lane == lane.id && car != &object) {
            if(cmp(car, &object)) {
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

    for(size_t i=0; i < s.getNumLights(); i++) {
        RedTrafficLight_id *light = s.getLight(i + s.getNumCars());
        if(light->lane == lane.id && light != &object) {
            if(cmp(light, &object)) {
                // objects behind
                if(closest_lt == nullptr || !cmp(light, closest_lt)) {
                    closest_lt = light;
                }
            } else {
                // objects in front of
                if(closest_gt == nullptr || cmp(light, closest_gt)) {
                    closest_gt = light;
                }
            }
        }
    }

    if(closest_gt != nullptr) {
        result.front = closest_gt->id;
    }
    if(closest_lt != nullptr) {
        result.back = closest_lt->id;
    }

    return result;
}


CUDA_HOSTDEV Road_id::NeighboringLanes AlgorithmWrapper::getNeighboringLanes(const Lane_id &lane) {
    Road_id::NeighboringLanes lanes;
    if (lane.lane_num > 0)
        lanes.left = s.getRoad(lane.road)->lanes[lane.lane_num  - 1];
    if (3 > lane.lane_num + 1)
        lanes.right = s.getRoad(lane.road)->lanes[lane.lane_num + 1];
    return lanes;
}



CUDA_HOSTDEV double AlgorithmWrapper::getAcceleration(TrafficObject_id &trafficObject, TrafficObject_id *leading_vehicle) {
    if (trafficObject.id < s.getNumCars()) {
        Car_id &car = s.assertCar(trafficObject);
        double vel_fraction = (car.v / C_MIN(s.getRoad(s.getLane(car.lane)->road)->limit, car.target_velocity));
        double without_lead = 1. - vel_fraction * vel_fraction * vel_fraction * vel_fraction; // faster than pow

        double with_lead = 0;
        if (leading_vehicle != nullptr) {
            double delta_v = car.v - leading_vehicle->v;
            double s = C_MAX(leading_vehicle->x - car.x - leading_vehicle->length, Car_id::min_s);
            with_lead = (car.min_distance + car.v * car.target_headway +
                         (car.v * delta_v) / (2. * sqrt(car.max_acceleration * car.target_deceleration))) / s;
            with_lead = with_lead * with_lead; // faster than pow
        }
        double acceleration = car.max_acceleration * (without_lead - with_lead);
        return acceleration;
    }
    return 0;
}


CUDA_HOSTDEV double AlgorithmWrapper::laneChangeMetric(Car_id &car, Lane_id::NeighboringObjects ownNeighbors, Lane_id::NeighboringObjects otherNeighbors) {

    if ((otherNeighbors.front == (size_t )-1 || (s.getTrafficObject(otherNeighbors.front)->x - car.x) >= (car.length / 2)) &&
        (otherNeighbors.back == (size_t ) -1 || (car.x - s.getTrafficObject(otherNeighbors.back)->x) >= (car.length / 2) + car.min_distance)) {
        double own_wo_lc = getAcceleration(car, s.getTrafficObject(ownNeighbors.front));
        double own_w_lc = getAcceleration(car, s.getTrafficObject(otherNeighbors.front));

        double other_lane_diff = 0;
        if (otherNeighbors.back != (size_t )-1) {
            other_lane_diff = getAcceleration(*s.getTrafficObject(otherNeighbors.back), &car) -
                              getAcceleration(*s.getTrafficObject(otherNeighbors.back), s.getTrafficObject(otherNeighbors.front));
        }


        double behind_diff = 0;
        if (ownNeighbors.back != (size_t )-1) {
            behind_diff = getAcceleration(*s.getTrafficObject(ownNeighbors.back), s.getTrafficObject(ownNeighbors.front)) -
                          getAcceleration(*s.getTrafficObject(ownNeighbors.back), &car);
        }

        if (own_w_lc > own_wo_lc) {
            return own_w_lc - own_wo_lc + car.politeness * (behind_diff + other_lane_diff);
        }
    }
    return 0;
}


CUDA_HOSTDEV double AlgorithmWrapper::getLaneChangeMetricForLane(TrafficObject_id &trafficObject, Lane_id *neighboringLane, const Lane_id::NeighboringObjects &ownNeighbors) {
    Car_id &car = s.assertCar(trafficObject);
    if (neighboringLane != nullptr) {
        Lane_id::NeighboringObjects neighbors = getNeighboringObjects(trafficObject, *neighboringLane);
        return laneChangeMetric(car, ownNeighbors, neighbors);
    }
    return 0;
}


CUDA_HOSTDEV Car_id::AdvanceData AlgorithmWrapper::nextStep(Car_id &car) {
    Lane_id::NeighboringObjects ownNeighbors = getNeighboringObjects(car, *s.getLane(car.lane));
    Road_id::NeighboringLanes neighboringLanes = getNeighboringLanes(*s.getLane(car.lane));

    double m_left = getLaneChangeMetricForLane(car, s.getLane(neighboringLanes.left), ownNeighbors);
    double m_right = getLaneChangeMetricForLane(car, s.getLane(neighboringLanes.right), ownNeighbors);


    if (m_left > 1 && m_left >= m_right) {
        // go to left lane
        return Car_id::AdvanceData(car.id, getAcceleration(car, s.getTrafficObject(getNeighboringObjects(car, *s.getLane(neighboringLanes.left)).front)), -1);
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        return Car_id::AdvanceData(car.id, getAcceleration(car, s.getTrafficObject(getNeighboringObjects(car, *s.getLane(neighboringLanes.right)).front)), 1);
    }
    else {
        // stay on lane
        return Car_id::AdvanceData(car.id, getAcceleration(car, s.getTrafficObject(ownNeighbors.front)), 0);
    }
}


CUDA_HOSTDEV Car_id::AdvanceData AlgorithmWrapper::nextStep(Car_id &car, Lane_id::NeighboringObjects leftNeighbors,
        Lane_id::NeighboringObjects ownNeighbors,Lane_id::NeighboringObjects rightNeighbors) {
    Road_id::NeighboringLanes neighboringLanes = getNeighboringLanes(*s.getLane(car.lane));
    // assert(ownNeighbors.front == getNeighboringObjects(car, *s.getLane(car.lane)).front);
    // assert(ownNeighbors.back == getNeighboringObjects(car, *s.getLane(car.lane)).back);

    double m_left = (neighboringLanes.left == (size_t ) -1) ? 0 : laneChangeMetric(car, ownNeighbors, leftNeighbors);
    // assert(m_left == getLaneChangeMetricForLane(car, s.getLane(neighboringLanes.left), ownNeighbors));
    double m_right = (neighboringLanes.right == (size_t ) -1) ? 0 : laneChangeMetric(car, ownNeighbors, rightNeighbors);
    // assert(m_right == getLaneChangeMetricForLane(car, s.getLane(neighboringLanes.right), ownNeighbors));


    if (m_left > 1 && m_left >= m_right) {
        // go to left lane
        double acc = getAcceleration(car, s.getTrafficObject(leftNeighbors.front));
        // assert(acc == getAcceleration(car, s.getTrafficObject(getNeighboringObjects(car, *s.getLane(neighboringLanes.left)).front)));

        return Car_id::AdvanceData(car.id, acc, -1);
    }
    else if (m_right > 1 && m_left < m_right) {
        // right go to right lane
        double acc = getAcceleration(car, s.getTrafficObject(rightNeighbors.front));
        // assert(acc == getAcceleration(car, s.getTrafficObject(getNeighboringObjects(car, *s.getLane(neighboringLanes.right)).front)));
        return Car_id::AdvanceData(car.id, acc, 1);
    }
    else {
        // stay on lane
        return Car_id::AdvanceData(car.id, getAcceleration(car, s.getTrafficObject(ownNeighbors.front)), 0);
    }
}


CUDA_HOSTDEV void AlgorithmWrapper::moveCarAcrossJunction(Car_id &car, Car_id::AdvanceData &data) {
    assert(car.turns_count != 0);

    Lane_id &old_lane = *s.getLane(car.lane);
    Road_id &road = *s.getRoad(old_lane.road);

    car.lane = (size_t) -1; // important to enforce ordering of lanes!

    // subtract moved position on current lane from distance
    car.x = (car.x - road.length);

    // select direction based on current direction and turn
    int direction = (road.roadDir + *s.getTurn(car.turns_begin + car.current_turn_offset) + 2) % 4;

    // if no road in that direction -> select next to the right
    size_t nextRoad;
    while ((nextRoad = s.getJunction(road.to)->outgoing[direction]) == (size_t )-1) direction = (++direction) % 4;

    // move car to same or the right lane AFTER lane change
    int8_t indexOfNextLane = C_MIN(2, (int8_t)old_lane.lane_num + data.lane_offset);
    indexOfNextLane = C_MAX((int8_t)0, indexOfNextLane);
    while(s.getRoad(nextRoad)->lanes[indexOfNextLane] == (size_t ) -1) indexOfNextLane--;
    car.lane = s.getRoad(nextRoad)->lanes[indexOfNextLane];

    // update next turns
    car.current_turn_offset = (car.current_turn_offset + 1) % car.turns_count;
}


CUDA_HOSTDEV void AlgorithmWrapper::updateKinematicState(Car_id &car, Car_id::AdvanceData &data) {
    assert(data.car == car.id);
    car.a = data.acceleration;
    car.v = C_MAX(car.v + car.a, 0.);
    car.x = (car.x + car.v);
}



CUDA_HOSTDEV void AlgorithmWrapper::advanceStep(Car_id &car, Car_id::AdvanceData &data) {
    updateKinematicState(car, data);
    updateLane(car, data);
}



CUDA_HOSTDEV void AlgorithmWrapper::updateLane(Car_id &car, Car_id::AdvanceData &data) {
    assert(data.car == car.id);
    // check for junction
    if (car.x > s.getLane(car.lane)->length) {
        moveCarAcrossJunction(car, data);
    }
    else {
        // just do a lane change if wanted
        if (data.lane_offset != 0) {
            // lane_offset should be validated in this case
            assert(3 > (s.getLane(car.lane)->lane_num + data.lane_offset));
            car.lane = s.getRoad(s.getLane(car.lane)->road)->lanes[s.getLane(car.lane)->lane_num + data.lane_offset];
        }
    }
}

CUDA_HOSTDEV void AlgorithmWrapper::setSignals(Junction_id &junction) {
    if (junction.signal_count == 0)
        return;
    for(int i = 0; i < 4; i++) {
        for(size_t &light_id : junction.red_traffic_lights_ids[i]) {
            if(light_id != (size_t ) -1) {
                RedTrafficLight_id &l = *s.getLight(light_id);
                if (s.getSignal(junction.signal_begin + junction.current_signal_id)->direction == i) {
                    // green light
                    l.lane = (size_t) -1;
                } else {
                    // red light
                    l.lane = l.mAssociatedLane;
                }
            }
        }
    }
}

CUDA_HOSTDEV void AlgorithmWrapper::updateSignals(Junction_id &junction) {
    if(junction.signal_count != 0 && 0 == --junction.current_signal_time_left) {
        junction.current_signal_id = ++junction.current_signal_id % junction.signal_count;
        junction.current_signal_time_left = s.getSignal(junction.signal_begin + junction.current_signal_id)->duration;
        setSignals(junction);
    }
}


CudaScenario_id CudaScenario_id::fromScenarioData(ScenarioData_id &scenario) {
    CudaScenario_id cudaScenario;
    cudaScenario.junctions = scenario.junctions.data();
    cudaScenario.signals = scenario.signals.data();
    cudaScenario.traffic_lights = scenario.traffic_lights.data();
    cudaScenario.roads = scenario.roads.data();
    cudaScenario.lanes = scenario.lanes.data();
    cudaScenario.cars = scenario.cars.data();
    cudaScenario.turns = scenario.turns.data();

    cudaScenario.num_cars = scenario.cars.size();
    cudaScenario.num_junctions = scenario.junctions.size();
    cudaScenario.num_lanes = scenario.lanes.size();
    cudaScenario.num_lights = scenario.traffic_lights.size();
    cudaScenario.num_roads = scenario.roads.size();
    cudaScenario.num_signals = scenario.signals.size();
    cudaScenario.num_turns = scenario.turns.size();
    return cudaScenario;
}

