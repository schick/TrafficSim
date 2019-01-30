//
// Created by oke on 07.12.18.
//

#include "model/Road.h"
#include <stdexcept>
#include <model/Car.h>
#include "model/Lane.h"
#include "model/Junction.h"

#include <assert.h>

inline void clearNeighbors(Car *car) {
    car->rightNeighbors.back = nullptr;
    car->rightNeighbors.front = nullptr;
    car->leftNeighbors.back = nullptr;
    car->leftNeighbors.front = nullptr;
    car->sameNeighbors.back = nullptr;
    car->sameNeighbors.front = nullptr;
}

inline void initNeighbors(Car *car, Car *copyFrom) {
    car->rightNeighbors.back = (copyFrom->rightNeighbors.back != nullptr &&
            Car::_Cmp()(copyFrom->rightNeighbors.back, car) ? copyFrom->rightNeighbors.back : nullptr);
    car->rightNeighbors.front = copyFrom->rightNeighbors.front;

    car->leftNeighbors.back = (copyFrom->leftNeighbors.back != nullptr &&
            Car::_Cmp()(copyFrom->leftNeighbors.back, car) ? copyFrom->leftNeighbors.back : nullptr);
    car->leftNeighbors.front = copyFrom->leftNeighbors.front;

    car->sameNeighbors.back = nullptr;
    car->sameNeighbors.front = nullptr;
}

void Road::preCalcNeighbors() {
    if (lanes.empty()) return;

    size_t num_lanes = lanes.size();

    // do not use dynamic memory
    std::array<const std::vector<Car *>*, 3> lane_cars {
        &lanes[0 % num_lanes]->getCars(),
        &lanes[1 % num_lanes]->getCars(),
        &lanes[2 % num_lanes]->getCars()
    };

    // do not use dynamic memory
    std::array<int, 3> lane_back = {
            (int) lane_cars[0]->size() - 1, (int) lane_cars[1]->size() - 1, (int) lane_cars[2]->size() - 1
    };

    size_t total_count = 0;

    for(int i = 0; i < num_lanes; i++) {
        total_count += lane_back[i] + 1;
        if (lane_back[i] + 1 > 0) clearNeighbors(lane_cars[i]->back());
    }

    while(total_count > 0) {
        int max_idx = -1;
        Car *max_car = nullptr;
        for(int i = 0; i < num_lanes; i++) {
            if(lane_back[i] != -1) {
                Car *current_car = (*lane_cars[i])[lane_back[i]];
                if (max_idx == -1 || Car::_Cmp()(max_car, current_car)) {
                    max_idx = i;
                    max_car = current_car;
                }
            }
        }

        if(max_idx > 0 && lane_back[max_idx - 1] != -1) {
            Car *leftBack = (*lane_cars[max_idx - 1])[lane_back[max_idx - 1]];
            max_car->leftNeighbors.back = leftBack;
            leftBack->rightNeighbors.front = max_car;
        }
        if(max_idx < num_lanes - 1 && lane_back[max_idx + 1] != -1) {
            Car *rightBack = (*lane_cars[max_idx + 1])[lane_back[max_idx + 1]];
            max_car->rightNeighbors.back = rightBack;
            rightBack->leftNeighbors.front = max_car;
        }

        if(lane_back[max_idx] - 1 != -1)
            initNeighbors((*lane_cars[max_idx])[lane_back[max_idx] - 1], max_car);

        if(lane_back[max_idx] > 0) {
            Car *ownBack = (*lane_cars[max_idx])[lane_back[max_idx] - 1];
            max_car->sameNeighbors.back = ownBack;
            ownBack->sameNeighbors.front = max_car;
        }

        lane_back[max_idx]--;
        total_count--;
    }

    /*
    Lane::NeighboringObjects empty;
    for(auto &l : lanes) {
        for(auto &c : l->getCars()) {
            Road::NeighboringLanes neighboringLanes = getNeighboringLanes(l);
            assert(c->sameNeighbors == l->getNeighboringObjects(c));
            assert(c->leftNeighbors == (neighboringLanes.left == nullptr ? empty : neighboringLanes.left->getNeighboringObjects(c)));
            assert(c->rightNeighbors == (neighboringLanes.right == nullptr ? empty : neighboringLanes.right->getNeighboringObjects(c)));
        }
    }*/
}


Road::NeighboringLanes Road::getNeighboringLanes(Lane* lane) {
    NeighboringLanes lanes;
    if (lane->lane > 0)
        lanes.left = lane->road.lanes[lane->lane - 1];
    if (lane->road.lanes.size() > lane->lane + 1)
        lanes.right = lane->road.lanes[lane->lane + 1];
    return lanes;
}

Junction::Direction Road::getDirection() {
    return roadDir;
}