
include_directories(include)

# by_id library
SET(SOURCE_FILES_BY_ID src/Car_id.cpp src/Junction.cpp src/Lane.cpp src/RedTrafficLight_id.cpp
        src/Road_id.cpp src/Scenario.cpp src/TrafficObject.cpp
        src/algorithms/SequentialAlgorithm_id.cpp
        src/algorithms/OpenMPAlgorithm_id.cpp
        src/algorithms/CudaAlgorithm.cpp)

IF(OpenCV_FOUND)
    set(SOURCE_FILES_BY_ID ${SOURCE_FILES_BY_ID} src/Visualization.cpp)
ENDIF(OpenCV_FOUND)

# For compilation ...
# Specify target & source files to compile it from
find_package(CUDA 9.0 REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++11)


IF(WITH_CUDA)
    add_definitions(-DCUDA)
    # TODO: add cuda files
    ELSE()
    message(WARNING "No cuda")
ENDIF()

cuda_add_library(cuda_container src/cudacontainer.cu)

add_library(scenario_id ${SOURCE_FILES_BY_ID})

target_link_libraries(scenario_id cuda_container)

target_link_libraries(scenario_id scenario_base)