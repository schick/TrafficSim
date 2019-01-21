
include_directories(include)

# add library with all logic
SET(SOURCE_FILES
        include/model/Car.h
        src/model/Car.cpp
        include/model/Junction.h
        src/model/Junction.cpp
        include/model/Lane.h
        src/model/Lane.cpp
        include/model/RedTrafficLight.h
        src/model/RedTrafficLight.cpp
        include/model/Road.h
        src/model/Road.cpp
        include/model/Scenario.h
        src/model/Scenario.cpp
        include/model/TrafficObject.h
        src/model/TrafficObject.cpp
        include/algorithms/SequentialAlgorithm.h
        src/algorithms/SequentialAlgorithm.cpp
        include/algorithms/OpenMPAlgorithm.h
        src/algorithms/OpenMPAlgorithm.cpp
        include/IntelligentDriverModel.h
        src/IntelligentDriverModel.cpp
        include/NeighborFinder.h
        src/NeighborFinder.cpp
        include/LaneChangeModel.h
        src/LaneChangeModel.cpp
        include/optimization/model/OptimizeScenario.h
        src/optimization/model/OptimizeScenario.cpp
        include/optimization/DistributionOpenMPOptimizer.h
        src/optimization/DistributionOpenMPOptimizer.cpp
        include/optimization/DistributionOptimizer.h
        src/optimization/DistributionOptimizer.cpp
        include/optimization/RandomOptimizer.h
        src/optimization/RandomOptimizer.cpp)

if(OpenCV_FOUND)
    set(SOURCE_FILES ${SOURCE_FILES} src/Visualization.cpp)
endif()

add_library(scenario_ref ${SOURCE_FILES})

link_libraries(scenario_ref scenario_base)

if(OpenCV_FOUND)
    target_link_libraries(scenario_ref ${OpenCV_LIBS})
endif()

