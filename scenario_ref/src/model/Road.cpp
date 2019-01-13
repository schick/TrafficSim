//
// Created by oke on 07.12.18.
//

#include "model/Road.h"
#include <stdexcept>
#include "model/Lane.h"
#include "model/Junction.h"

Road::NeighboringLanes Road::getNeighboringLanes(Lane* lane) {
    NeighboringLanes lanes;
    if (lane->lane > 0)
        lanes.left = lane->road->lanes[lane->lane - 1];
    if (lane->road->lanes.size() > lane->lane + 1)
        lanes.right = lane->road->lanes[lane->lane + 1];
    return lanes;
}

Junction::Direction Road::getDirection() {
    return roadDir;
}