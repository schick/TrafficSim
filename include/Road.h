//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_ROAD_H
#define PROJECT_ROAD_H


#include <vector>
#include "Junction.h"
class Lane;


class Road {
public:

    struct NeighboringLanes {
        Lane *right = nullptr;
        Lane *left = nullptr;
    };

    Road(Junction *_from, Junction *_to, double _limit) : from(_from), to(_to), limit(_limit / 3.6) {};

    /**
     * properties
     */
    Junction *from;
    Junction *to;
    double limit;

    std::vector<Lane*> lanes;
    /**
     * must return first left than right.
     * @param lane
     * @return
     */
    NeighboringLanes getNeighboringLanes(Lane* lane);


    /**
     * get length of road
     * @return length of road in m
     */
    double getLength();

    Junction::Direction getDirection();
};



#endif //PROJECT_ROAD_H
