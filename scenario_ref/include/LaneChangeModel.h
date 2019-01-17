//
// Created by maxi on 1/8/19.
//

#ifndef TRAFFIC_SIM_LANECHANGEMODEL_H
#define TRAFFIC_SIM_LANECHANGEMODEL_H


#include <model/Car.h>
#include <model/Lane.h>
#include <IntelligentDriverModel.h>

class LaneChangeModel {

public:

    /**
    * lane change metric described on slide 19 (22)
    * @param car the relevant car
    * @param neighboringLane the other lane
    * @param ownNeighbors neighbors on current lane
    * @param otherNeighbors neighbors on other lane
    * @return metric value in m/s^2
    */
    static double getLaneChangeMetric(Car &car, Lane::NeighboringObjects &ownNeighbors, Lane *otherLane, Lane::NeighboringObjects &otherNeighbors, bool isLeftLane);

private:

    static double calculateLaneChangeMetric(Car &car, Lane::NeighboringObjects &sameNeighbors, Lane::NeighboringObjects &otherNeighbors, bool isLeftLane);
    static bool hasFrontSpaceOnOtherLane(Car &car, Lane::NeighboringObjects &otherNeighbors);
    static bool hasBackSpaceOnOtherLane(Car &car, Lane::NeighboringObjects &otherNeighbors);

};


#endif //TRAFFIC_SIM_LANECHANGEMODEL_H
