
include_directories(include)

SET(SOURCE_FILES_S_BASE
        src/AdvanceAlgorithm.cpp
        src/BaseOptimizer.cpp)

add_library(scenario_base ${SOURCE_FILES_S_BASE})

IF(OpenCV_FOUND)
    target_link_libraries(scenario_base ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)