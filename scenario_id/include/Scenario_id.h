
#ifndef PROJECT_SCENARIO_ID_H
#define PROJECT_SCENARIO_ID_H

#include "ScenarioData_id.h"
#include "BaseScenario.h"

using json = nlohmann::json;

class Scenario_id : public ScenarioData_id, public BaseScenario {
public:

    void parse(json &input);
    void initJunctions();
    void parseCars(json &input);
    void parseRoads(json &input);
    void parseJunctions(json &input);
    json toJson();
};

#endif