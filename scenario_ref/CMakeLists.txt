
include_directories(include)

# add library with all logic

SET(SOURCE_FILES
        include/model/Car.h
        src/model/Car.cpp
        include/model/Junction.h
        src/model/Junction.cpp
        include/model/Lane.h
        src/model/Lane.cpp
        include/model/TrafficLight.h
        src/model/TrafficLight.cpp
        include/model/Road.h
        src/model/Road.cpp
        include/model/Scenario.h
        src/model/Scenario.cpp
        include/model/TrafficObject.h
        include/algorithms/SequentialAlgorithm.h
        src/algorithms/SequentialAlgorithm.cpp
        include/algorithms/OpenMPAlgorithm.h
        src/algorithms/OpenMPAlgorithm.cpp
        include/optimization/model/OptimizeScenario.h
        src/optimization/model/OptimizeScenario.cpp
        include/optimization/DistributionOptimizer.h
        src/optimization/DistributionOptimizer.cpp
        include/optimization/ParallelRandomOptimizer.h
        src/optimization/ParallelRandomOptimizer.cpp
        include/optimization/model/SignalLayout.h
        src/optimization/model/SignalLayout.cpp
        src/optimization/GeneticOptimizer.cpp
        include/optimization/GeneticOptimizer.h
        src/optimization/SequentialRandomOptimizer.cpp
        include/optimization/SequentialRandomOptimizer.h
        include/algorithms/RefAlgorithm.h)

if(OpenCV_FOUND)
    set(SOURCE_FILES ${SOURCE_FILES} src/Visualization.cpp)
endif()

add_library(scenario_ref ${SOURCE_FILES})

link_libraries(scenario_ref scenario_base)

if(OpenCV_FOUND)
    target_link_libraries(scenario_ref ${OpenCV_LIBS})
endif()

