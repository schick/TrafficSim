#!/usr/bin/env bash

rm CMakeFiles/ cmake_install.cmake libtraffic_scenario.a traffic_sim libvis.a Makefile -R -f

# use cin mode and disable visualization
cmake . -G "Unix Makefiles"

rm CMakeFiles/ cmake_install.cmake libtraffic_scenario.a traffic_sim libvis.a CMakeCache.txt -R -f
