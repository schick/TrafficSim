//
// Created by maxi on 12/16/18.
//

#include "util/SimpleArgumentParser.h"

SimpleArgumentParser &SimpleArgumentParser::add_argument(const char *name) {
    args_names.push_back(name);
    return *this;
}

SimpleArgumentParser &SimpleArgumentParser::add_kw_argument(const char *name, const char *default_) {
    long idx = std::distance(kwargs_names.begin(), std::find(kwargs_names.begin(), kwargs_names.end(), std::string(name)));
    if (idx < kwargs_names.size()) {
        std::string &a = kwargs.at((size_t )idx);
        a = default_;
    } else {
        kwargs.emplace_back(default_);
        kwargs_names.emplace_back(std::string(name));
    }
    return *this;
}

bool SimpleArgumentParser::load(int argc, char *argv[]) {
    if (argc <= args_names.size()) {
        printf("Not enough arguments");
        return false;
    }
    int i = 0;

    for (; i < args_names.size(); i++) {
        args.emplace_back(argv[i + 1]);
#ifdef DEBUG_MSGS
        printf("Argument: %s := %s\n", args_names[i].c_str(), args[i].c_str());
#endif
    }

    std::string name;
    while (i + 2 < argc) {
        name = argv[i + 1] + 1;
        auto idx = std::distance(kwargs_names.begin(), std::find(kwargs_names.begin(), kwargs_names.end(), name));
        if (idx >= kwargs_names.size()) {
            printf("No such argument: %s", name.c_str());
            return false;
        }
        kwargs[idx] = argv[i + 2];
#ifdef DEBUG_MSGS
        printf("Argument: %s := %s\n", kwargs_names[idx].substr(1).c_str(), kwargs[idx].c_str());
#endif
        i += 2;
    }
    return true;
}

std::string SimpleArgumentParser::operator[](const std::string &key) {
    std::string name = key;
    auto idx = std::distance(kwargs_names.begin(), std::find(kwargs_names.begin(), kwargs_names.end(), name));
    if (idx < kwargs_names.size())
        return kwargs[idx];
    idx = std::distance(args_names.begin(), std::find(args_names.begin(), args_names.end(), key));
    if (idx < kwargs_names.size())
        return args[idx];
    assert(false && "Invalid Argument");
}
