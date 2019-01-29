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

    Road(Junction *from, Junction *to, double limit, Junction::Direction roadDir) : from(from), to(to), limit(limit), roadDir(roadDir) {};

    /**
     * must return first left than right.
     * @param lane
     * @return
     */
    NeighboringLanes getNeighboringLanes(Lane* lane);

    /**
     * get the direction of the road
     * @return the direction
     */
    Junction::Direction getDirection();

    /**
     * properties
     */
    Junction *from;
    Junction *to;
    double limit;

    std::vector<Lane*> lanes;

    Junction::Direction roadDir;

};

#endif //PROJECT_ROAD_H
