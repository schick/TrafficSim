//
// Created by oke on 15.12.18.
//

#ifndef PROJECT_ADVANCEALGORITHM_H
#define PROJECT_ADVANCEALGORITHM_H

#include "Scenario.h"

// registering based on: https://stackoverflow.com/questions/11175379/register-a-c-class-so-that-later-a-function-can-iterate-over-all-registered-cl
class AdvanceAlgorithm {
public:

    virtual ~AdvanceAlgorithm() = default;

    using create_f = std::shared_ptr<AdvanceAlgorithm>(Scenario *scenario);

    static void registrate(std::string const & name, create_f * fp);

    static std::shared_ptr<AdvanceAlgorithm> instantiate(std::string const & name, Scenario* scenario);

    template <typename D>
    struct Registrar
    {
        explicit Registrar(std::string const & name)
        {
            AdvanceAlgorithm::registrate(name, &D::create);
        }
        // make non-copyable, etc.
    };


    Scenario *getScenario() { return scenario; }

    explicit AdvanceAlgorithm(Scenario *scenario) : scenario(scenario) {};
    virtual void advance(size_t steps) = 0;

    static std::vector<std::string> getAlgorithms() {
        std::vector<std::string> names;
        for(auto kv : registry()) {
            names.push_back(kv.first);
        }
        return names;
    }

private:

    static std::unordered_map<std::string, create_f *> & registry();

    Scenario *scenario;

};

#define STR(str) #str
#define REGISTER_ALGORITHM(CLASS) AdvanceAlgorithm::Registrar<CLASS> CLASS##_registar(STR(CLASS))

#endif //PROJECT_ADVANCEALGORITHM_H
