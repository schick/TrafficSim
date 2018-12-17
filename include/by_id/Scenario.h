
#include "util/json.hpp"
#include "by_id/ScenarioData.h"

using json = nlohmann::json;

class Scenario : public ScenarioData {
public:
    void parse(json input);
    void initJunctions();
    void parseCars(json & input);
    void parseRoads(json & input);
    void createRoads(const nlohmann::json & road);
    void createLanesForRoad(const nlohmann::json & road, int road_id);
    void parseJunctions(json &input);
    json toJson();
};