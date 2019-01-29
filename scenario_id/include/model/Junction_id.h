//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_JUNCTION_ID_H
#define PROJECT_JUNCTION_ID_H

#include <stddef.h>

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
        size_t duration;
        Direction direction;
    };

    Junction_id() : id((size_t)-1), x((size_t)-1), y((size_t)-1), signal_begin((size_t)-1), signal_count(0),
                 current_signal_id((size_t)-1), current_signal_time_left((size_t)-1),
                 incoming(), outgoing(), red_traffic_lights_ids()  {
        for(auto &array : red_traffic_lights_ids) for(auto &i : array) i = (size_t ) -1;
        for(size_t &i : incoming) i = (size_t) -1;
        for(size_t &i : outgoing) i = (size_t) -1;
    }

    Junction_id(size_t id, double x, double y, size_t signal_begin, size_t signal_count) :
            id(id), x(x), y(y), signal_begin(signal_begin), signal_count(signal_count),
            current_signal_id((size_t)-1), current_signal_time_left((size_t)-1),
            incoming(), outgoing(), red_traffic_lights_ids() {
        for(auto &array : red_traffic_lights_ids) for(auto &i : array) i = (size_t ) -1;
        for(size_t &i : incoming) i = (size_t) -1;
        for(size_t &i : outgoing) i = (size_t) -1;
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

    size_t outgoing[4];
    size_t incoming[4];

    /**
     * red lights for each lane and road.
     */
    size_t red_traffic_lights_ids[4][3];

    /**
     * current signal
     */
    size_t current_signal_id;

    /**
     * time left in current signal
     */
    size_t current_signal_time_left;

};

#endif //PROJECT_JUNCTION_H
