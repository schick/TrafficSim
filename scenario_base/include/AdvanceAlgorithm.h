//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_ADVANCEALGORITHM_H
#define PROJECT_ADVANCEALGORITHM_H

#include <memory>
#include <vector>
#include <unordered_map>

#include "BaseScenario.h"
#include "util/json_fwd.hpp"
#include "BaseVisualizationEngine.h"


#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif


using json = nlohmann::json;

// registering based on: https://stackoverflow.com/questions/11175379/register-a-c-class-so-that-later-a-function-can-iterate-over-all-registered-cl
class AdvanceAlgorithm {
public:

    // constructor type for a AdvanceAlgorithm
    using create_f = std::shared_ptr<AdvanceAlgorithm>(std::shared_ptr<BaseScenario> scenario);
    // constructor type for a scenario ontop which this algorithm works
    using create_scenario_f = std::shared_ptr<BaseScenario>();

    // register a factory class with given name, AdvanceAlgorithm and BaseScenario constructors
    static void registrate(std::string const & name, create_f * fp, create_scenario_f * sc);

    // create a AdvanceAlgorithm instance based on given name
    static std::shared_ptr<AdvanceAlgorithm> instantiate(std::string const & name, std::shared_ptr<BaseScenario> scenario);
    static std::shared_ptr<AdvanceAlgorithm> instantiate(std::string const & name, json &scenario);

    // create a visualization that works with the BaseScenario for this algorithm
    virtual std::shared_ptr<BaseVisualizationEngine> createVisualizationEngine(std::shared_ptr<BaseScenario> &scenario) { return nullptr; }

    // register class that registers this.
    template <typename D>
    struct Registrar
    {
        explicit Registrar(std::string const & name)
        {
            AdvanceAlgorithm::registrate(name, &D::create, &D::createScenario);
        }
        // make non-copyable, etc.
    };

    std::shared_ptr<BaseScenario> &getScenario() { return scenario; }

    explicit AdvanceAlgorithm(std::shared_ptr<BaseScenario> &scenario) : scenario(scenario) {};
    virtual void advance(size_t steps) = 0;

    static std::vector<std::string> getAlgorithms() {
        std::vector<std::string> names;
        for(auto kv : registry()) {
            names.push_back(kv.first);
        }
        return names;
    }

private:

    typedef struct {
        AdvanceAlgorithm::create_f *creater;
        AdvanceAlgorithm::create_scenario_f *scenario_creator;
    } creator_struct;

    // static registry
    static std::unordered_map<std::string, AdvanceAlgorithm::creator_struct> & registry();

    std::shared_ptr<BaseScenario> scenario;

};

#define STR(str) #str
#define REGISTER_ALGORITHM(CLASS) static AdvanceAlgorithm::Registrar<CLASS> CLASS##_registar(STR(CLASS))

#ifdef VISUALIZATION_ENABLED
#define ADVANCE_ALGO_INIT(CLASS_TYPE, SCENARIO_TYPE, VISUALIZATION_TYPE) \
    static std::shared_ptr<AdvanceAlgorithm> create(std::shared_ptr<BaseScenario> scenario) { return std::make_shared<CLASS_TYPE>(scenario); } \
    static std::shared_ptr<BaseScenario> createScenario() { return std::make_shared<SCENARIO_TYPE>(); }\
      std::shared_ptr<BaseVisualizationEngine> createVisualizationEngine(std::shared_ptr<BaseScenario> &scenario) override { return std::make_shared<VISUALIZATION_TYPE>(scenario); }
#else
#define ADVANCE_ALGO_INIT(CLASS_TYPE, SCENARIO_TYPE, VISUALIZATION_TYPE) \
    static std::shared_ptr<AdvanceAlgorithm> create(std::shared_ptr<BaseScenario> scenario) { return std::make_shared<CLASS_TYPE>(scenario); } \
    static std::shared_ptr<BaseScenario> createScenario() { return std::make_shared<SCENARIO_TYPE>(); }
#endif

#endif //PROJECT_ADVANCEALGORITHM_H
