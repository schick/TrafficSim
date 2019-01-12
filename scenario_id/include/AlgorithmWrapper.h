//
// Created by oke on 22.12.18.
//

#ifndef TRAFFIC_SIM_ALGORITHMWRAPPER_H
#define TRAFFIC_SIM_ALGORITHMWRAPPER_H

#include "Junction_id.h"
#include "Road_id.h"
#include "Car_id.h"
#include "Scenario_id.h"
#include <assert.h>

template<typename T>
class CArrayIterator {

    T *array;
    size_t len;
public:
    CUDA_HOSTDEV CArrayIterator(T *array, size_t len) : array(array), len(len) {}

    CUDA_HOSTDEV T *begin() {
        return array;
    }

    CUDA_HOSTDEV T *end() {
        return array + len;
    }
};

class CudaScenario_id {

public:
    static CudaScenario_id *fromScenarioData_device(ScenarioData_id &scenario);
    static CudaScenario_id fromScenarioData(ScenarioData_id &scenario);
    static void freeDeviceCudaScenario(CudaScenario_id *device_cuda_scenario);
    void retriveData(Scenario_id *scenario);

    CUDA_HOSTDEV inline Junction_id *getJunction(size_t id) {
        if(id >= num_junctions) return nullptr;
        return junctions + id;
    }

    CUDA_HOSTDEV inline CArrayIterator<Junction_id> getJunctionIterator() {
        return CArrayIterator<Junction_id>(junctions, num_junctions);
    }

    CUDA_HOSTDEV inline Car_id *getCar(size_t id) {
        if(id >= num_cars) return nullptr;
        return cars + id;
    }

    CUDA_HOSTDEV inline CArrayIterator<Car_id> getCarIterator() {
        return CArrayIterator<Car_id>(cars, num_cars);
    }

    CUDA_HOSTDEV inline TrafficObject_id *getTrafficObject(size_t id) {
        if(id >= num_cars + num_lights) return nullptr;
        if (id < num_cars)
            return cars + id;
        else
            return traffic_lights + (id - num_cars);
    }

    CUDA_HOSTDEV inline RedTrafficLight_id *getLight(size_t id) {
        if(!(id >= num_cars && id < num_cars + num_lights)) return nullptr;
        return traffic_lights + (id - num_cars);
    }

    CUDA_HOSTDEV inline Road_id *getRoad(size_t id) {
        if(id >= num_roads) return nullptr;
        return roads + id;
    }

    CUDA_HOSTDEV inline CArrayIterator<Lane_id> getLaneIterator() {
        return CArrayIterator<Lane_id>(lanes, num_lanes);
    }

    CUDA_HOSTDEV inline Lane_id *getLane(size_t id) {
        if(id >= num_lanes) return nullptr;
        return lanes + id;
    }

    CUDA_HOSTDEV inline Car_id::TurnDirection *getTurn(size_t id) {
        if(id >= num_turns) return nullptr;
        return turns + id;
    }

    CUDA_HOSTDEV inline Junction_id::Signal *getSignal(size_t id) {
        if(id >= num_signals) return nullptr;
        return signals + id;
    }

    CUDA_HOSTDEV inline size_t getNumLights() {
        return num_lights;
    }

    CUDA_HOSTDEV inline size_t getNumCars() {
        return num_cars;
    }
    CUDA_HOSTDEV inline size_t getNumJunctions() {
        return num_junctions;
    }
    CUDA_HOSTDEV inline size_t getNumLanes() {
        return num_lanes;
    }

    CUDA_HOSTDEV inline Car_id &assertCar(TrafficObject_id &trafficObject) {
        assert(trafficObject.id < num_cars);
        return reinterpret_cast<Car_id&>(trafficObject);
    }

private:
    // junctions.
    Junction_id* junctions;
    size_t num_junctions;
    Junction_id::Signal* signals;
    size_t num_signals;
    RedTrafficLight_id* traffic_lights;
    size_t num_lights;

    // roads
    Road_id* roads;
    size_t num_roads;
    Lane_id* lanes;
    size_t num_lanes;

    // cars
    Car_id* cars;
    size_t num_cars;
    Car_id::TurnDirection* turns;
    size_t num_turns;

public:
    typedef Road_id Road;
    typedef TrafficObject_id TrafficObject;
    typedef Lane_id Lane;
    typedef Car_id Car;
};


class AlgorithmWrapper {
    CudaScenario_id &s;
public:
    CUDA_HOSTDEV explicit AlgorithmWrapper(CudaScenario_id &s) : s(s) {}
    CUDA_HOSTDEV inline CudaScenario_id &getScenario(){ return s; };

    CUDA_HOSTDEV Lane_id::NeighboringObjects getNeighboringObjects(const TrafficObject_id &object, Lane_id &lane);
    CUDA_HOSTDEV Road_id::NeighboringLanes getNeighboringLanes(const Lane_id &lane);
    CUDA_HOSTDEV double getAcceleration(TrafficObject_id &car, TrafficObject_id *leading_vehicle);
    CUDA_HOSTDEV double laneChangeMetric(Car_id &car, Lane_id::NeighboringObjects ownNeighbors,
                                         Lane_id::NeighboringObjects otherNeighbors);
    CUDA_HOSTDEV double laneChangeMetric(Car_id &car, Lane_id::NeighboringObjectsRef ownNeighbors,
                                         Lane_id::NeighboringObjectsRef otherNeighbors);
    CUDA_HOSTDEV double getLaneChangeMetricForLane(TrafficObject_id &trafficObject,
                                      Lane_id *neighboringLane,
                                      const Lane_id::NeighboringObjects &ownNeighbors);
    CUDA_HOSTDEV Car_id::AdvanceData nextStep(Car_id &car);
    CUDA_HOSTDEV Car_id::AdvanceData nextStep(Car_id &car, Lane_id::NeighboringObjects leftNeighbors,
                                              Lane_id::NeighboringObjects ownNeighbors,Lane_id::NeighboringObjects rightNeighbors);
    CUDA_HOSTDEV Car_id::AdvanceData nextStep(Car_id &car, Lane_id::NeighboringObjectsRef leftNeighbors,
                                              Lane_id::NeighboringObjectsRef ownNeighbors,Lane_id::NeighboringObjectsRef rightNeighbors);

    CUDA_HOSTDEV void moveCarAcrossJunction(Car_id &car, Car_id::AdvanceData &data);
    CUDA_HOSTDEV void updateKinematicState(Car_id &car, Car_id::AdvanceData &data);
    CUDA_HOSTDEV void advanceStep(Car_id &car, Car_id::AdvanceData &data);
    CUDA_HOSTDEV void updateLane(Car_id &car, Car_id::AdvanceData &data);
    CUDA_HOSTDEV void updateSignals(Junction_id &junction);
    CUDA_HOSTDEV void setSignals(Junction_id &junction);
};

#endif //TRAFFIC_SIM_ALGORITHMWRAPPER_H
