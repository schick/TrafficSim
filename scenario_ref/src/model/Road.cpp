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
    std::vector<int> lane_back(lanes.size());

    std::vector<std::vector<Car *>> lane_cars;
    lane_cars.reserve(lanes.size());

    size_t total_count = 0;

    for(int i = 0; i < lanes.size(); i++) {
        lane_cars.emplace_back(lanes[i]->getCars());
        lane_back[i] = (int) lane_cars[i].size() - 1;
        total_count += lane_cars[i].size();
        if (lane_cars[i].size() > 0) clearNeighbors(lane_cars[i].back());
    }

    while(total_count > 0) {
        int max_idx = -1;
        Car *max_car = nullptr;
        for(int i = 0; i < lanes.size(); i++) {
            if(lane_back[i] != -1) {
                Car *current_car = lane_cars[i][lane_back[i]];
                if (max_idx == -1 || Car::_Cmp()(max_car, current_car)) {
                    max_idx = i;
                    max_car = current_car;
                }
            }
        }

        if(max_idx > 0 && lane_back[max_idx - 1] != -1) {
            Car *leftBack = lane_cars[max_idx - 1][lane_back[max_idx - 1]];
            max_car->leftNeighbors.back = leftBack;
            leftBack->rightNeighbors.front = max_car;
        }
        if(max_idx < lanes.size() - 1 && lane_back[max_idx + 1] != -1) {
            Car *rightBack = lane_cars[max_idx + 1][lane_back[max_idx + 1]];
            max_car->rightNeighbors.back = rightBack;
            rightBack->leftNeighbors.front = max_car;
        }

        if(lane_back[max_idx] - 1 != -1)
            initNeighbors(lane_cars[max_idx][lane_back[max_idx] - 1], max_car);

        if(lane_back[max_idx] > 0) {
            Car *ownBack = lane_cars[max_idx][lane_back[max_idx] - 1];
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
            assert(c->ownNeighbors == l->getNeighboringObjects(c));
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