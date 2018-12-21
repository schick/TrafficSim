//
// Created by oke on 16.12.18.
//

#include "Car_id.h"
#include "algorithms/CudaAlgorithm_id.h"


void CudaAlgorithm_id::advance(size_t steps) {

    Scenario_id *scenario_id = getIDScenario();
    auto &cars = scenario_id->cars;
    auto &junctions = scenario_id->junctions;
    auto &lights = scenario_id->traffic_lights;

    std::vector<TrafficObject_id> to_cars(cars.size());
    for(int i=0; i < cars.size(); i++) to_cars[i] = cars[i];
    TrafficObject_id *device_compare_traffic_objects = cars_container.createDeviceObjects(to_cars);

    std::vector<Car_id::AdvanceData> changes(cars.size());
    {
        std::vector<size_t> own_front(cars.size()), own_back(cars.size()), left_front(cars.size()), left_back(cars.size()), right_front(cars.size()), right_back(cars.size());

        for (int i = 0; i < steps; i++) {

            cars_container.get_nearest_objects_hd(device_compare_traffic_objects, left_front, left_back, -1);
            cars_container.get_nearest_objects_hd(device_compare_traffic_objects, right_front, right_back, 1);
            cars_container.get_nearest_objects_hd(device_compare_traffic_objects, own_front, own_back, 0);
            #pragma omp parallel for
            for (int i = 0; i < cars.size(); i++) {
                changes[i] = cars[i].nextStep(*scenario_id, own_front[i], own_back[i], left_front[i], left_back[i], right_front[i], right_back[i]);
            }

            #pragma omp parallel for
            for (int i = 0; i < changes.size(); i++) {
                scenario_id->cars[changes[i].car].advanceStep(*scenario_id, changes[i]);
            }

            #pragma omp parallel for
            for (int i = 0; i < junctions.size(); i++) {
                junctions[i].updateSignals(*scenario_id);
            }

            std::vector<TrafficObject_id> &container_objects = cars_container.get_w_objects();
            container_objects.resize(cars.size());
            for(size_t i=0; i < cars.size(); i++) container_objects[i] = cars[i];
            cars_container.setDeviceObjects(device_compare_traffic_objects, container_objects);
            container_objects.resize(cars.size() + lights.size());
            for(size_t i=0; i < lights.size(); i++) container_objects[i + cars.size()] = lights[i];
        }
    }
}