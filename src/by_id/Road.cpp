//
// Created by oke on 07.12.18.
//

#include <stdexcept>

#include "by_id/Road.h"
#include "by_id/Lane.h"
#include "by_id/Junction.h"
#include "by_id/Scenario.h"

Road::NeighboringLanes Road::getNeighboringLanes(Scenario &s, Lane &lane) {
    NeighboringLanes lanes;
    if (lane.lane_num > 0)
        lanes.left = lane.id - 1;
    if (s.roads.at(lane.road).lanes.size() > lane.lane_num + 1)
        lanes.right = s.roads.at(lane.road).lanes[lane.lane_num  + 1];
    return lanes;
}