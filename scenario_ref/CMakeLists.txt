
include_directories(include)

# add library with all logic
SET(SOURCE_FILES src/Car.cpp src/Junction.cpp src/Lane.cpp src/RedTrafficLight.cpp
        src/Road.cpp src/Scenario.cpp src/TrafficObject.cpp
        src/algorithms/SequentialAlgorithm.cpp
        src/algorithms/OpenMPAlgorithm.cpp
        src/InteligentDriverModel.cpp)

if(OpenCV_FOUND)
    set(SOURCE_FILES ${SOURCE_FILES} src/Visualization.cpp)
endif()

add_library(scenario_ref ${SOURCE_FILES})

link_libraries(scenario_ref scenario_base)

if(OpenCV_FOUND)
    target_link_libraries(scenario_ref ${OpenCV_LIBS})
endif()

