struct signal {
    size_t dir;
    size_t time;
};

struct junction {
    size_t idx;
    size_t x;
    size_t y;
    signal signals[4];
};

struct road {
    size_t to_id;
    size_t from_id;
    size_t from_x;
    size_t from_y;
    size_t to_x;
    size_t to_y;
    size_t lanes;
    size_t length;
    size_t limit;
};
struct car {
    float target_deceleration;
    float max_acceleration;
    float target_headway;
    float politeness;
    int route[4];
    float target_velocity;
    int id;
    float min_distance;
    int from;
    int to;
    int lane;
    int distance;

};


int get_random(int argc, char *argv[], std::vector<car> &host_cars, std::vector<junction> &host_junctions, std::vector<road> &host_roads);