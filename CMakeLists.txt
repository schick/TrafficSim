cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)


option(WITHOUT_VIS "Disable Visualization to dismiss opencv dependency" ON)
option(USE_CIN "Use cin to read JSON" ON)

# enable optimization in release mode
string(TOLOWER "${CMAKE_BUILD_TYPE}" DEBUG_TYPE_LOWER )
if(DEBUG_TYPE_LOWER STREQUAL debug)
else()
    message(STATUS "release mode...")
    set(CMAKE_CXX_FLAGS -O3)
endif()


if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

# Add include directory to include search path
include_directories(include)

# add library for visualization
if(USE_CIN)
    add_definitions(-DUSE_CIN)
endif()

if(WITHOUT_VIS)
    message(WARNING "visualization disabled.")
else()
    # find opencv for visualization
    find_package( OpenCV QUIET)
    IF(OpenCV_FOUND)
        add_definitions(-DVISUALIZATION_ENABLED)
        add_library(vis src/Visualization.cpp)
        target_link_libraries(vis ${OpenCV_LIBS})
    ELSE()
        message(WARNING "OpenCV not found. Building without visualization.")
    ENDIF(OpenCV_FOUND)
endif()


# add library with all logic
add_library(traffic_scenario src/Junction.cpp src/Car.cpp src/RedTrafficLight.cpp
        src/Lane.cpp src/Road.cpp src/TrafficObject.cpp src/Scenario.cpp)

# create final executable
add_executable(traffic_sim src/main.cpp)
target_link_libraries(traffic_sim traffic_scenario)

IF (OpenCV_FOUND)
    target_link_libraries(traffic_sim vis)
ENDIF()