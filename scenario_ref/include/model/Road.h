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

    Road(Junction *from, Junction *to, double limit, Junction::Direction roadDir) : from(from), to(to), limit(limit), roadDir(roadDir) {
        lenght = (double) (abs(from->x - to->x) + abs(from->y - to->y));
    };

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
     * get the length of the road
     * @return length of road in m
     */
    double getLength();

    /**
     * get the direction of the road
     * @return the direction
     */
    Junction::Direction getDirection();

private:
    Junction::Direction roadDir;

    double lenght;
};



#endif //PROJECT_ROAD_H
