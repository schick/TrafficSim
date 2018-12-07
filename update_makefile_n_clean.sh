#!/usr/bin/env bash

rm CMakeFiles/ cmake_install.cmake libtraffic_scenario.a traffic_sim libvis.a CMakeCache.txt Makefile -R -f

# use cin mode and disable visualization
cmake . -DWITHOUT_VIS=ON -DUSE_CIN=ON -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"

rm CMakeFiles/ cmake_install.cmake libtraffic_scenario.a traffic_sim libvis.a -R -f
