//
// Created by oke on 07.12.18.
//

#include "Road.h"
#include <stdexcept>
#include <Lane.h>
#include <Junction.h>

std::vector<Lane*> Road::getNeighboringLanes(Lane* lane) {
    std::vector<Lane*> lanes;
    if (lane->lane_id > 0)
        lanes.push_back(lane->road->lanes[lane->lane_id - 1]);
    if (lane->road->lanes.size() > lane->lane_id + 1)
        lanes.push_back(lane->road->lanes[lane->lane_id + 1]);
    return lanes;
}


double Road::getLength() {
    return (abs(from->x - to->x) + abs(from->y - to->y));
}


Junction::Direction Road::getDirection() {
    // linkshändisches koordinatensystem
    if (from->y < to->y) {
        return Junction::Direction::SOUTH;
    } else if (from->y > to->y) {
        return Junction::Direction::NORTH;
    } else if (from->x < to->x) {
        return Junction::Direction::EAST;
    } else if (from->x > to->x) {
        return Junction::Direction::WEST;
    }
    printf("ERROR: not a valid road...");
    exit(-1);
}