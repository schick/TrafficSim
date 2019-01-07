cmake_minimum_required(VERSION 2.8)
project(traffic_sim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

option(VISUALIZATION "Disable Visualization to dismiss opencv dependency" OFF)
option(USE_CIN "Use cin to read JSON" ON)
option(DEBUG_MSGS "Print stuff" OFF)
option(BUILD_TESTS "Build Tests" OFF)
option(WITH_CUDA "use cuda" ON)
option(ALL_ALGOS "use all algorithsm - not only preffered" OFF)

option(OKE "Oke's default config" OFF)
IF(OKE)
    set(VISUALIZATION ON)
    set(USE_CIN OFF)
    set(DEBUG_MSGS ON)
    set(BUILD_TESTS ON)
    set(WITH_CUDA ON)
ENDIF()

IF(ALL_ALGOS)
    add_definitions(-DALL_ALGORITHMS)
ENDIF()

IF(WITH_CUDA) # enable cuda
    enable_language(CUDA)
ENDIF()

if(DEBUG_MSGS) # enable debugging in source code
    add_definitions(-DDEBUG_MSGS)
ENDIF()

# enable optimization in release mode
string(TOLOWER "${CMAKE_BUILD_TYPE}" DEBUG_TYPE_LOWER)
if(DEBUG_TYPE_LOWER STREQUAL debug)
    add_definitions(-DDEBUG)
else()
    message(STATUS "building in release mode")
    set(CMAKE_CXX_FLAGS -O3) #TODO: Visual Studio?
endif()

# add library for visualization
if(USE_CIN)
    add_definitions(-DUSE_CIN)
endif()

# load openmp
find_package(OpenMP QUIET)
if (OPENMP_FOUND)
    message(STATUS "building with openmp")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "building without openmp")
endif()

# find threads lib the cross-plattform way
find_package (Threads QUIET)
IF(Threads_FOUND)
ELSE()
    error("Threads not found")
ENDIF()

# check for opencv
if(VISUALIZATION)
    find_package(OpenCV QUIET)
    IF(OpenCV_FOUND)
        add_definitions(-DVISUALIZATION_ENABLED)
    ELSE()
        message(WARNING "OpenCV not found. Building without visualization.")
    ENDIF(OpenCV_FOUND)
endif()

# build libraries
include_directories(scenario_base/include scenario_ref/include scenario_id/include)
add_subdirectory(scenario_base)
add_subdirectory(scenario_ref)
add_subdirectory(scenario_id)

# create final executable
include_directories(include)


add_executable(traffic_sim src/main.cpp src/util/SimpleArgumentParser.cpp)

IF(WITH_CUDA)
    set_property(TARGET traffic_sim PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(traffic_sim ${CUDA_LIBRARIES})
ENDIF()

target_link_libraries(traffic_sim scenario_ref scenario_id scenario_base)

IF(OpenCV_FOUND)
    target_link_libraries(traffic_sim ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)

# build tests & benchmark
IF (BUILD_TESTS)
    message(STATUS "building tests")
    include(GoogleTest)
    add_executable(json_test src/test/test_json.cpp)
    target_link_libraries(json_test gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(json_test scenario_ref scenario_id )
    gtest_discover_tests(json_test)

    add_executable(benchmark src/test/benchmark.cpp)
    target_link_libraries(benchmark scenario_ref scenario_id scenario_base)
ELSE()
    message(STATUS "not building tests")
ENDIF()


add_executable(compare src/compare_algorithms.cpp)
target_link_libraries(compare scenario_id scenario_base scenario_ref ${CUDA_LIBRARIES} ${OpenCV_LIBS})