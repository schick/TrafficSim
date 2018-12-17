//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_JUNCTION_H
#define PROJECT_JUNCTION_H

#include <array>
#include <vector>

#include "by_id/RedTrafficLight.h"

class Scenario;
class Road;

/**
 * a junction...
 */
class Junction {

public:
    /**
     * direction to describe and access incoming and outgoing roads
     */
    enum Direction {
        NORTH = 0,
        EAST = 1,
        SOUTH = 2,
        WEST = 3
    };

    /**
     * traffic light signal
     */
    struct Signal {
        Signal(int duration, Direction direction) : duration(duration), direction(direction) {}
        int duration;
        Direction direction;
    };

    Junction() : id(-1), x(-1), y(-1), signal_begin(-1), signal_count(0),
                 current_signal_id(-1), current_signal_time_left(-1), incoming(), outgoing()  {
        for(auto &array : red_traffic_lights_id) array.fill(-1);
        incoming.fill(-1);
        outgoing.fill(-1);
    }

    Junction(int id, double x, double y, int signal_begin, int signal_count) :
            id(id), x(x), y(y), signal_begin(signal_begin), signal_count(signal_count),
            current_signal_id(-1), current_signal_time_left(-1), incoming(), outgoing() {
        for(auto &array : red_traffic_lights_id)
            array.fill(-1);
        incoming.fill(-1);
        outgoing.fill(-1);
    };

    /**
     * signals to cycle through
     */
    int signal_begin;
    int signal_count;

    /**
     * properties
     */
    int id;
    double x, y;
    std::array<int, 4> outgoing;
    std::array<int, 4> incoming;
    std::array<std::array<int, 3>, 4> red_traffic_lights_ids;

    /**
     * initialize signals before starting the algorithm. create RedTrafficLight-Objects.
     */
    void initializeSignals(Scenario &s);

    /**
     * update signals. called each timestep.
     */
    void updateSignals(Scenario &s);

    /**
     * current signal
     */
    int current_signal_id;
private:


    /**
     * time left in current signal
     */
    int current_signal_time_left;

    /**
     * red lights for each lane and road.
     */
    std::array<std::array<int, 3>, 4> red_traffic_lights_id;

    void setSignals(Scenario &s);
};




#endif //PROJECT_JUNCTION_H
