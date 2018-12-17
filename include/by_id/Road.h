//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_ROAD_H
#define PROJECT_ROAD_H

#include "by_id/Junction.h"

class Road {
public:

    /**
     * properties
     */
    int id;
    int from;
    int to;
    double limit;
    double length;
    std::array<int, 3> lanes;
    Junction::Direction roadDir;


    struct NeighboringLanes {
        int right = -1;
        int left = -1;
    };

    Road(int id, int from_id, int to_id, double limit, double length, Junction::Direction roadDir) :
            id(id), from(from_id), to(to_id), length(length), limit(limit), roadDir(roadDir), lanes() {
        lanes.fill(-1);
    };
    Road() : id(-1), from(-1), to(-1), length(0), limit(0), roadDir(Junction::Direction::NORTH), lanes() {
        lanes.fill(-1);
    };

    /**
     * must return first left than right.
     * @param lane
     * @return
     */
    NeighboringLanes getNeighboringLanes(Scenario &s, Lane &lane);

};



#endif //PROJECT_ROAD_H
