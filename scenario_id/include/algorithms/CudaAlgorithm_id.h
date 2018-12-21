//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_CUDAALGORITHM_ID_H
#define PROJECT_CUDAALGORITHM_ID_H

#include <memory>

#include "AdvanceAlgorithm.h"
#include "Car_id.h"
#include "Scenario_id.h"
#include "Visualization_id.h"
#include "cudacontainer.h"

class CudaAlgorithm_id : public AdvanceAlgorithm {

public:
    ADVANCE_ALGO_INIT(CudaAlgorithm_id, Scenario_id, Visualization_id);

    explicit CudaAlgorithm_id(std::shared_ptr<BaseScenario> scenario) : AdvanceAlgorithm(scenario),
            cars_container(getIDScenario()->cars.size() + getIDScenario()->traffic_lights.size(), getIDScenario()->lanes.size()) {

        _initializeLanes();
        std::vector<Car_id> &cars = getIDScenario()->cars;
        std::vector<RedTrafficLight_id> &traffic_lights = getIDScenario()->traffic_lights;
        std::vector<TrafficObject_id> &objects = cars_container.get_rw_objects();
        for(int i=0; i < cars.size(); i++) {
            objects[i] = cars[i];
        }
        for(int i=0; i < traffic_lights.size(); i++) {
            objects[i + cars.size()] = traffic_lights[i];
        }

        /*
        std::vector<TrafficObject_id> objects_cpy = cars_container.get_objects();
        test_right_lane_change(objects_cpy);
        test_left_lane_change(objects_cpy);*/

    }

    Scenario_id* getIDScenario() {
        return dynamic_cast<Scenario_id*>(getScenario().get());
    }

    void advance(size_t steps) override;

private:

    SortedUniqueObjectContainer cars_container;

    void _initializeLanes() {
        std::vector<Lane_id> lanes = getIDScenario()->lanes;
        std::vector<size_t> rightLaneId(lanes.size());
        std::vector<size_t> leftLaneId(lanes.size());
        for(int i=0; i < lanes.size(); i++) {
            Road_id::NeighboringLanes neigbors = (getIDScenario()->
                    roads[getIDScenario()->lanes[i].road]
                    .getNeighboringLanes(*getIDScenario(), getIDScenario()->lanes[i]));
            rightLaneId[i] = neigbors.right;
            leftLaneId[i] = neigbors.left;
        }
        cars_container.setLaneInformation(leftLaneId, rightLaneId);
    }

    void test_left_lane_change(const std::vector<TrafficObject_id> &test_objs) {
        std::vector<size_t> front(test_objs.size()), back(test_objs.size());
        const std::vector<TrafficObject_id> &container_objects = cars_container.get_objects();

        cars_container.get_nearest_objects_host(test_objs, front, back, -1);
        for (size_t i = 0; i < test_objs.size(); i++) {
            if (test_objs[i].getLane() == -1)
                continue;
            const Lane_id &l = getIDScenario()->lanes[test_objs[i].getLane()];
            const Road_id &r = getIDScenario()->roads[l.road];
            Road_id::NeighboringLanes neighboringLanes = r.getNeighboringLanes(*getIDScenario(), l);

            if (neighboringLanes.left == (size_t) -1) {
                assert(front.at(i) == (size_t) -1);
                assert(back.at(i) == (size_t) -1);
                continue;
            }

            Lane_id &left_lane = getIDScenario()->lanes[neighboringLanes.left];
            Lane_id::NeighboringObjects neig = left_lane.getNeighboringObjects(*getIDScenario(), test_objs[i]);

            if (neig.back == -1) {
                assert(back.at(i) == (size_t) -1);
            } else {
                assert(back.at(i) != (size_t) -1 && container_objects.at(back.at(i)).id == neig.back);
            }
            if (neig.front == -1) {
                assert(front.at(i) == (size_t) -1);
            } else {
                assert(front.at(i) != (size_t) -1 && container_objects.at(front.at(i)).id == neig.front);
            }
        }
        printf("passed left lc test.\n");
    }

    void test_right_lane_change(const std::vector<TrafficObject_id> &test_objs) {
        std::vector<size_t> front(test_objs.size()), back(test_objs.size());
        const std::vector<TrafficObject_id> &container_objects = cars_container.get_objects();

        cars_container.get_nearest_objects_host(test_objs, front, back, 1);
        for (size_t i = 0; i < test_objs.size(); i++) {
            if (test_objs[i].getLane() == -1)
                continue;
            const Lane_id &l = getIDScenario()->lanes[test_objs[i].getLane()];
            const Road_id &r = getIDScenario()->roads[l.road];
            Road_id::NeighboringLanes neighboringLanes = r.getNeighboringLanes(*getIDScenario(), l);

            if (neighboringLanes.right == (size_t) -1) {
                assert(front.at(i) == (size_t) -1);
                assert(back.at(i) == (size_t) -1);
                continue;
            }

            Lane_id &left_lane = getIDScenario()->lanes[neighboringLanes.right];
            Lane_id::NeighboringObjects neig = left_lane.getNeighboringObjects(*getIDScenario(), test_objs[i]);

            if (neig.back == -1) {
                assert(back.at(i) == (size_t) -1);
            } else {
                assert(back.at(i) != (size_t) -1 && container_objects.at(back.at(i)).id == neig.back);
            }
            if (neig.front == -1) {
                assert(front.at(i) == (size_t) -1);
            } else {
                assert(front.at(i) != (size_t) -1 && container_objects.at(front.at(i)).id == neig.front);
            }
        }
        printf("passed right lc test.\n");
    }


};

#endif //PROJECT_SEQUENTIALALGORITHM_H
