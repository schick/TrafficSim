
cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS )# -O3)

set(CMAKE_CXX_STANDARD 17)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

# Add include directory to include search path
include_directories(include)

# find opencv for visualization
find_package( OpenCV REQUIRED )

# add library for visualization
add_library(vis src/Visualization.cpp)
target_link_libraries(vis ${OpenCV_LIBS})

# add library with all logic
add_library(traffic_scenario src/Junction.cpp src/Car.cpp src/RedTrafficLight.cpp
        src/Lane.cpp src/Road.cpp src/TrafficObject.cpp src/Scenario.cpp)

# create final executable
add_executable(traffic_sim src/main.cpp)
target_link_libraries(traffic_sim vis traffic_scenario)