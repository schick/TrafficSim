//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_ROAD_ID_H
#define PROJECT_ROAD_ID_H

#include "Junction_id.h"

class Road_id {
public:

    /**
     * properties
     */
    size_t id;
    size_t from;
    size_t to;
    double limit;
    double length;
    std::array<size_t , 3> lanes;
    Junction_id::Direction roadDir;


    struct NeighboringLanes {
        size_t right = (size_t) -1;
        size_t left = (size_t) -1;
    };

    Road_id(size_t id, size_t from_id, size_t to_id, double limit, double length, Junction_id::Direction roadDir) :
            id(id), from(from_id), to(to_id), length(length), limit(limit), roadDir(roadDir), lanes() {
        lanes.fill((size_t )-1);
    };
    Road_id() : id((size_t )-1), from((size_t )-1), to((size_t )-1), length(0), limit(0), roadDir(Junction_id::Direction::NORTH), lanes() {
        lanes.fill((size_t )-1);
    };

    /**
     * must return first left than right.
     * @param lane
     * @return
     */
    NeighboringLanes getNeighboringLanes(const Scenario_id &s, const Lane_id &lane) const;

};



#endif //PROJECT_ROAD_H
