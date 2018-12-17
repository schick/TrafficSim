//
// Created by maxi on 12/16/18.
//
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>

#ifndef PROJECT_SIMPLEARGUMENTPARSER_H
#define PROJECT_SIMPLEARGUMENTPARSER_H


class SimpleArgumentParser {
private:
    std::vector<std::string> args_names;
    std::vector<std::string> kwargs_names;

    std::vector<std::string> args;
    std::vector<std::string> kwargs;

public:

    SimpleArgumentParser &add_argument(const char* name);

    SimpleArgumentParser &add_kw_argument(const char* name, const char* default_);

    bool load(int argc, char* argv[]);

    std::string operator[] (const std::string &key);
};


#endif //PROJECT_SIMPLEARGUMENTPARSER_H
