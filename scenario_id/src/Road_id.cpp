//
// Created by oke on 07.12.18.
//

#include <stdexcept>

#include "Road_id.h"
#include "Lane_id.h"
#include "Junction_id.h"
#include "Scenario_id.h"

Road_id::NeighboringLanes Road_id::getNeighboringLanes(const Scenario_id &s, const Lane_id &lane) const {
    NeighboringLanes lanes;
    if (lane.lane_num > 0)
        lanes.left = lane.id - 1;
    if (s.roads.at(lane.road).lanes.size() > lane.lane_num + 1)
        lanes.right = s.roads.at(lane.road).lanes[lane.lane_num  + 1];
    return lanes;
}