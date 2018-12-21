//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_JUNCTION_ID_H
#define PROJECT_JUNCTION_ID_H

#include <array>
#include <vector>

#include "RedTrafficLight_id.h"

class Scenario_id;
class Road_id;

/**
 * a junction...
 */
class Junction_id {

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

    Junction_id() : id((size_t)-1), x((size_t)-1), y((size_t)-1), signal_begin((size_t)-1), signal_count(0),
                 current_signal_id((size_t)-1), current_signal_time_left((size_t)-1), incoming(), outgoing()  {
        for(auto &array : red_traffic_lights_id) array.fill((size_t)-1);
        incoming.fill((size_t)-1);
        outgoing.fill((size_t)-1);
    }

    Junction_id(size_t id, double x, double y, size_t signal_begin, size_t signal_count) :
            id(id), x(x), y(y), signal_begin(signal_begin), signal_count(signal_count),
            current_signal_id((size_t)-1), current_signal_time_left((size_t)-1), incoming(), outgoing() {
        for(auto &array : red_traffic_lights_id)
            array.fill((size_t)-1);
        incoming.fill((size_t)-1);
        outgoing.fill((size_t)-1);
    };

    /**
     * signals to cycle through
     */
    size_t signal_begin;
    size_t signal_count;

    /**
     * properties
     */
    size_t id;
    double x, y;
    std::array<size_t, 4> outgoing;
    std::array<size_t, 4> incoming;
    std::array<std::array<size_t, 3>, 4> red_traffic_lights_ids;

    /**
     * initialize signals before starting the algorithm. create RedTrafficLight-Objects.
     */
    void initializeSignals(Scenario_id &s);

    /**
     * update signals. called each timestep.
     */
    void updateSignals(Scenario_id &s);

    /**
     * current signal
     */
    size_t current_signal_id;
private:


    /**
     * time left in current signal
     */
    size_t current_signal_time_left;

    /**
     * red lights for each lane and road.
     */
    std::array<std::array<size_t, 3>, 4> red_traffic_lights_id;

    void setSignals(Scenario_id &s);
};




#endif //PROJECT_JUNCTION_H
