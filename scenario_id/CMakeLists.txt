
include_directories(include)

# by_id library
SET(SOURCE_FILES_BY_ID src/Car_id.cpp src/Junction.cpp src/Lane.cpp src/RedTrafficLight_id.cpp
        src/Road_id.cpp src/Scenario.cpp src/TrafficObject.cpp
        src/algorithms/SequentialAlgorithm_id.cpp
        src/algorithms/OpenMPAlgorithm_id.cpp)

IF(OpenCV_FOUND)
    set(SOURCE_FILES_BY_ID ${SOURCE_FILES_BY_ID} src/Visualization.cpp)
ENDIF(OpenCV_FOUND)

IF(WITH_CUDA)
    add_definitions(-DCUDA)
    set(SOURCE_FILES_BY_ID ${SOURCE_FILES_BY_ID}) # TODO: add cuda files
ENDIF()

add_library(scenario_id ${SOURCE_FILES_BY_ID})
target_link_libraries(scenario_id scenario_base)