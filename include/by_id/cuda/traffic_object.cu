

class TrafficObject {

public:

    /**
     * compare object to compare Traffic objects by
     */
    struct Cmp {
        bool operator () (const TrafficObject *lhs, TrafficObject *rhs) {
            if(lhs->x == rhs->x)
                return lhs->id > rhs->id;
            return lhs->x < rhs->x;
        }
    };

    TrafficObject(int id=-1, double length=0, double x=0, double v=0, double a=0) : id(id), x(x), v(v), a(a), length(length), lane(-1) {};

    /**
     * state. put acceleration in here for a more generic implementation of Car::nextStep
     */
    int id;
    double v;
    double a;
    double length;

    /**
    * calculate the acceleration with 'leading_vehicle' as lead
    * it will be assumed that 'leading_vehicle' is on current lane
    * @param leading_vehicle the leading vehicle
    * @return acceleration for t + 1
    */
    virtual double getAcceleration(Scenario &s, int leading_vehicle_id) { return 0; }

    /**
     * move a this object to a specific lane.
     * @param lane lane to move object to
     */
    void moveToLane(int lane_id);

    /**
     * remove object from any lane it may be assigned to
     */
    void removeFromLane();

    /**
     * get currently assigned lane
     * @return currently assigned lane
     */
    int getLane() const;

    double getPosition() const;
    void setPosition(double x) {
        this->x = x;
    }

    double x;

private:
    /**
     * current lane
     */
    int lane;

    void _moveToLane(Lane *lane);
    void _removeFromLane();

};



class Car : public TrafficObject {

private:

    /**
     * lane change metric described on slide 19 (22)
     * @param ownNeighbors neighbors on current lane
     * @param otherNeighbors neighbors on other lane
     * @return metric value in m/s^2
     */
    double laneChangeMetric(Scenario &s, Lane::NeighboringObjects ownNeighbors, Lane::NeighboringObjects otherNeighbors);

public:

    /**
     * Data representing next advance of a car
     */
    struct AdvanceData {
        AdvanceData() = default;
        AdvanceData(int car, double acceleration, int lane_offset)
                : car(car), acceleration(acceleration), lane_offset(lane_offset) {};
        int car = -1;
        double acceleration = 0;
        int8_t lane_offset = 0;
    };


    /**
     * data representing a turn at an intersection
     */
    enum TurnDirection {
        UTURN = 0,
        LEFT = 1,
        STRAIGHT = 2,
        RIGHT = 3
    };

    /**
     * some constructor
     * @param id
     * @param length
     * @param target_velocity
     * @param max_acceleration
     * @param target_deceleration
     * @param min_distance
     * @param target_headway
     * @param politeness
     * @param x
     * @param v
     * @param a
     */
    Car(int id=-1, double length=0, double target_velocity=0, double max_acceleration=0, double target_deceleration=0,
        double min_distance=0, double target_headway=0, double politeness=0,
        double x=0, double v=0, double a=0)
            : length(length), target_velocity(target_velocity), max_acceleration(max_acceleration),
              target_deceleration(target_deceleration), min_distance(min_distance),
              target_headway(target_headway), politeness(politeness), TrafficObject(id, length, x, v, a) {}

    /**
     * properties
     */


    double length;
    double target_velocity;
    double max_acceleration;
    double target_deceleration;
    double min_distance;
    //definition from ilias:
    static constexpr double min_s = 0.001;
    double target_headway;
    double politeness;


    std::list<TurnDirection> turns;

    /**
     * calculate advance-data for next step
     * @return data representing the change
     */
    AdvanceData nextStep(Scenario &s);


    double getLaneChangeMetricForLane(Scenario &s, int neighboringLane, const Lane::NeighboringObjects &ownNeighbors);

    /**
     * advance car based of data
     * @param data data representing the change
     */
    void advanceStep(Scenario &s, AdvanceData &data);

    /**
     * calculate the desired acceleration. base calculation on leading object
     * @param leading_object leading object. may actually be in a different lane, this methods treats every object
     *      passed with this parameter as if it where in current lane
     * @return acceleration in m/s^2
     */
    double getAcceleration(Scenario &s, int leading_object_id) override;

private:

    void updateLane(Scenario &s, AdvanceData &data);

    bool isCarOverJunction(Scenario &s);

    void moveCarAcrossJunction(Scenario &s, Car::AdvanceData &data);

    void updateKinematicState(Car::AdvanceData &data);
};
