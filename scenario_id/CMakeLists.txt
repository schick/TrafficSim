cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX)

include_directories(include)

# by_id library
SET(SOURCE_FILES_BY_ID
        include/model/Car_id.h
        include/model/Junction_id.h
        include/model/Lane_id.h
        include/model/RedTrafficLight_id.h
        include/model/Road_id.h
        include/model/Scenario_id.h
        src/model/Scenario_id.cpp
        include/model/TrafficObject_id.h)

IF(OpenCV_FOUND)
    set(SOURCE_FILES_BY_ID ${SOURCE_FILES_BY_ID} src/Visualization.cpp)
ENDIF(OpenCV_FOUND)

IF(WITH_CUDA)
    message(STATUS "with cuda")
    enable_language(CUDA)

    # For compilation ...
    # Specify target & source files to compile it from
    find_package(CUDA 9.0 REQUIRED)

    # Pass options to NVCC
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++14)
    list(APPEND CUDA_NVCC_FLAGS "-O3")

    add_definitions(-DWITH_CUDA)

    set(SOURCE_FILES_BY_ID ${SOURCE_FILES_BY_ID}
            include/algorithms/AlgorithmWrapper.h
            src/algorithms/AlgorithmWrapper.cu
            include/algorithms/CudaAlgorithm.h
            src/algorithms/CudaAlgorithm.cu
            include/cuda_utils/PreScan.h
            src/cuda_utils/PreScan.cu
            include/cuda_utils/cuda_utils.h
            src/cuda_utils/cuda_utils.cu
            include/cuda_utils/SortedBucketContainer.h
            src/cuda_utils/SortedBucketContainer.cu
            include/cuda_utils/SortBuffer.h
            src/cuda_utils/SortBuffer.cu)
ELSE()
    message(STATUS "WITHOUT cuda")
ENDIF()

add_library(scenario_id ${SOURCE_FILES_BY_ID})

IF(WITH_CUDA)
    set_property(TARGET scenario_id PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_features(scenario_id PUBLIC cxx_std_14)
    set_target_properties(scenario_id PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ENDIF()

target_link_libraries(scenario_id scenario_base)
