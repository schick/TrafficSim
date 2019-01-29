//
// Created by oke on 07.12.18.
//

#ifndef PROJECT_JUNCTION_H
#define PROJECT_JUNCTION_H

#include <array>
#include <vector>
#include <atomic>

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
        Signal() : duration(0), direction(NORTH) {};
        Signal(uint64_t duration, Direction direction) : duration(duration), direction(direction) {}
        uint64_t duration;
        Direction direction;
    };

    Junction(uint64_t id, double x, double y);
    Junction(const Junction &other);

    /**
     * initialize signals before starting the algorithm. create RedTrafficLight-Objects.
     */
    void initializeSignals();

    /**
     * update signals. called each timestep.
     */
    void updateSignals();

    /**
     * Return a vector containing all possible Direction
     * @return the direction vector
     */
    std::vector<Direction> getPossibleDirections();


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

    std::array<std::atomic<uint64_t >, 4> incoming_counter;

private:

    /**
     * time left in current signal
     */
    uint64_t current_signal_time_left = 0;

    void setSignals();
};

#endif //PROJECT_JUNCTION_H
