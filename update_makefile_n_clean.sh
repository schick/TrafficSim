#!/usr/bin/env bash

rm CMakeFiles/ cmake_install.cmake libtraffic_scenario.a traffic_sim libvis.a Makefile "json_test[1]_include.cmake" -R -f

# use cin mode and disable visualization
cmake . -G "Unix Makefiles"

# replace absolute paths
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
sed -i "s?${DIR}?.?g" ./Makefile 

# cleanup
rm CMakeFiles/ cmake_install.cmake libtraffic_scenario.a traffic_sim libvis.a CMakeCache.txt "json_test[1]_include.cmake" -R -f
