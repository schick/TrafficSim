//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_ROAD_H
#define PROJECT_ROAD_H


#include <vector>


class Junction;
class Lane;


class Road {
public:
    Road(Junction *_from, Junction *_to, double _limit) : from(_from), to(_to), limit(_limit) {};

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
    std::vector<Lane*> getNeighboringLanes(Lane* lane);


    /**
     * get length of road
     * @return length of road in m
     */
    double getLength();
};



#endif //PROJECT_ROAD_H
