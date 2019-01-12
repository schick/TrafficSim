cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX)

include_directories(include)

# by_id library
SET(SOURCE_FILES_BY_ID src/Car_id.cpp src/Junction.cpp src/Lane.cpp src/RedTrafficLight_id.cpp
        src/Road_id.cpp src/Scenario.cpp src/TrafficObject.cpp
        src/algorithms/SequentialAlgorithm_id.cpp
        src/algorithms/OpenMPAlgorithm_id.cpp
        src/algorithms/CudaAlgorithm.cpp src/AlgorithmWrapper.cpp src/algorithms/SequentialCudaDataAlgorithm_id.cpp)

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
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++17)
    list(APPEND CUDA_NVCC_FLAGS "-O3")

    add_definitions(-DWITH_CUDA)

    set(SOURCE_FILES_BY_ID ${SOURCE_FILES_BY_ID}
            src/cudacontainer.cu
            src/AlgorithmWrapper.cu
            #src/algorithms/CudaAlgorithm2_id.cu
            src/algorithms/TestAlgo.cu)
ELSE()
    message(STATUS "WITHOUT cuda")
ENDIF()

add_library(scenario_id ${SOURCE_FILES_BY_ID})

IF(WITH_CUDA)
    set_property(TARGET scenario_id PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_features(scenario_id PUBLIC cxx_std_17)
    set_target_properties(scenario_id PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
ENDIF()

target_link_libraries(scenario_id scenario_base)
