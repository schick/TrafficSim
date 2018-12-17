//
// Created by oke on 07.12.18.
//

#include "Road.h"
#include <stdexcept>
#include "Lane.h"
#include "Junction.h"

Road::NeighboringLanes Road::getNeighboringLanes(Lane* lane) {
    NeighboringLanes lanes;
    if (lane->lane > 0)
        lanes.left = lane->road->lanes[lane->lane - 1];
    if (lane->road->lanes.size() > lane->lane + 1)
        lanes.right = lane->road->lanes[lane->lane + 1];
    return lanes;
}

double Road::getLength() {
    return (abs(from->x - to->x) + abs(from->y - to->y));
}

Junction::Direction Road::getDirection() {
    return roadDir;
}