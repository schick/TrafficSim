#!/usr/bin/env bash

# directory of this file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

# change into repository directory
cd ${DIR}

# files to delete for cleanup
CLEANUP_FILES="CMakeFiles/ cmake_install.cmake libtraffic_scenario.a traffic_sim libvis.a
        json_test[1]_include.cmake json_test[1]_tests.cmake"

# cleanup caches and old builds
rm ${CLEANUP_FILES} -R -f

# use cin mode and disable visualization
cmake . -G "Unix Makefiles"

# replace absolute paths
sed -i "s?${DIR}?.?g" ./Makefile 

# cleanup - we only want to keep the Makefile
rm ${CLEANUP_FILES} -R -f
