//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_JUNCTION_H
#define PROJECT_JUNCTION_H

#include <array>
#include <vector>

#include "TrafficLight.h"
#include "atomic"

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
        Signal() {};
        Signal(uint64_t duration, Direction direction) : duration(duration), direction(direction) {}
        uint64_t duration;
        Direction direction;
    };

    Junction(uint64_t id, double x, double y) : id(id), x(x), y(y), incoming(), outgoing(), incoming_counter() {};
    Junction(const Junction &other) :
            signals(other.signals), current_signal(other.current_signal), current_signal_time_left(other.current_signal_time_left),
            id(other.id), x(other.x), y(other.y),
            outgoing(other.outgoing), incoming(other.incoming), incoming_counter() {
        for(int i = 0; i < 4; i++) incoming_counter[i] = other.incoming_counter[i].load();
    }

    /**
     * signals to cycle through
     */
    std::vector<Signal> signals;

    /**
     * current signal
     */
    size_t current_signal = 0;

    /**
     * properties
     */
    uint64_t id;
    double x, y;
    std::array<Road*, 4> outgoing;
    std::array<Road*, 4> incoming;

    /**
     * initialize signals before starting the algorithm. create RedTrafficLight-Objects.
     */
    void initializeSignals();

    /**
     * update signals. called each timestep.
     */
    void updateSignals();

    std::array<std::atomic<uint64_t >, 4> incoming_counter;

    /**
     * Return a vector containing all possible Direction
     * @return the direction vector
     */
    std::vector<Direction> getPossibleDirections();

private:

    /**
     * time left in current signal
     */
    uint64_t current_signal_time_left = 0;

    void setSignals();
};




#endif //PROJECT_JUNCTION_H
