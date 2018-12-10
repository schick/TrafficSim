cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_STANDARD 17)

# Add include directory to include search path
include_directories(include)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

option(WITHOUT_VIS "Disable Visualization to dismiss opencv dependency" ON)
option(USE_CIN "Use cin to read JSON" ON)
option(BUILD_TESTS "Build Tests" OFF)

option(OKE "Oke's default config" OFF)

IF(OKE)
    set(WITHOUT_VIS OFF)
    set(USE_CIN OFF)
    set(BUILD_TESTS ON)
ENDIF()

# enable optimization in release mode
string(TOLOWER "${CMAKE_BUILD_TYPE}" DEBUG_TYPE_LOWER )
if(DEBUG_TYPE_LOWER STREQUAL debug)
else()
    message(STATUS "building in release mode")
    set(CMAKE_CXX_FLAGS -O3)
endif()
# add library for visualization
if(USE_CIN)
    add_definitions(-DUSE_CIN)
endif()

if(WITHOUT_VIS)
    message(STATUS "not building visualization")
else()
    # find opencv for visualization
    message(STATUS "building visualization")
    find_package(OpenCV QUIET)
    IF(OpenCV_FOUND)
        add_definitions(-DVISUALIZATION_ENABLED)
        add_library(vis src/Visualization.cpp)
        target_link_libraries(vis ${OpenCV_LIBS})
    ELSE()
        message(WARNING "OpenCV not found. Building without visualization.")
    ENDIF(OpenCV_FOUND)
endif()

# openmp
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    message(STATUS "building with openmp")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "building without openmp")
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

IF (BUILD_TESTS)
    message(STATUS "building tests")
    include(GoogleTest)
    add_executable(json_test src/test_json.cpp)
    target_link_libraries(json_test gtest gtest_main pthread)
    target_link_libraries(json_test traffic_scenario)
    gtest_discover_tests(json_test)
ELSE()
    message(STATUS "not building tests")
ENDIF()
