cmake_minimum_required(VERSION 2.8)
project(traffic_sim LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
endif(MSVC)

function(copy_test_files target)
    if (MSVC)
        add_custom_command(TARGET ${target} PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_CURRENT_BINARY_DIR}/tests)
    else(MSVC)
        add_custom_command(TARGET ${target} PRE_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_directory
                           ${CMAKE_SOURCE_DIR}/tests $<TARGET_FILE_DIR:${target}>/tests)
    endif(MSVC)
endfunction()

option(VISUALIZATION "Disable Visualization to dismiss opencv dependency" OFF)
option(USE_CIN "Use cin to read JSON" ON)
option(DEBUG_MSGS "Print stuff" OFF)
option(RUN_WITH_TESTS "Enable tests in each step. - slow." OFF)
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

IF(RUN_WITH_TESTS)
    add_definitions(-DRUN_WITH_TESTS)
ENDIF()

IF(ALL_ALGOS)
    add_definitions(-DALL_ALGORITHMS)
ENDIF()


find_package(CUDA)

if (CUDA_FOUND)
    #Get CUDA compute capability

    set(OUTPUTFILE ${CMAKE_CURRENT_SOURCE_DIR}/main/src/util/cuda_script) # No suffix required
    set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/main/src/util/cuda_info.cu)
    execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
    execute_process(COMMAND sleep 0.5)
    execute_process(COMMAND ${OUTPUTFILE} RESULT_VARIABLE CUDA_RETURN_CODE OUTPUT_VARIABLE ARCH)

    if(CUDA_RETURN_CODE EQUAL 0)
        set(CUDA_SUCCESS "TRUE")
    else()
        set(CUDA_SUCCESS "FALSE")
    endif()

    if (${CUDA_SUCCESS})
        message(STATUS "CUDA Architecture: ${ARCH}")
        message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
        message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
        message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")

        set(CUDA_NVCC_FLAGS "${ARCH}")
        add_definitions(-DGPU) #You may not require this

    else()
        message(WARNING ${ARCH})
    endif()
endif()

IF(WITH_CUDA) # enable cuda
    enable_language(CUDA OPTIONAL)
    if(CUDA_FOUND AND CMAKE_CUDA_COMPILER)
    else()
        set(WITH_CUDA OFF)
    endif()
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
include_directories(main/include)


add_executable(traffic_sim main/src/main.cpp main/src/util/SimpleArgumentParser.cpp main/src/TrafficSim.cpp)


IF(WITH_CUDA)
    set_property(TARGET traffic_sim PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set_property(TARGET traffic_sim PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    # target_link_libraries(traffic_sim ${CUDA_LIBRARIES})
    target_compile_features(traffic_sim PUBLIC cxx_std_14)

    add_definitions(-DWITH_CUDA)
ELSE()
ENDIF()

target_link_libraries(traffic_sim scenario_ref scenario_id scenario_base)

IF(OpenCV_FOUND)
    target_link_libraries(traffic_sim ${OpenCV_LIBS})
ENDIF(OpenCV_FOUND)
   
copy_test_files(traffic_sim)

# build tests & benchmark
IF (BUILD_TESTS)
    message(STATUS "building tests")
    include(GoogleTest)
    add_executable(json_test main/src/test/test_json.cpp)
    target_link_libraries(json_test gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(json_test scenario_ref scenario_id )
    gtest_discover_tests(json_test)
    copy_test_files(json_test)

    add_executable(benchmark main/src/test/benchmark.cpp)
    target_link_libraries(benchmark scenario_ref scenario_id scenario_base)
    copy_test_files(benchmark)

ELSE()
    message(STATUS "not building tests")
ENDIF()


add_executable(compare main/src/test/compare_algorithms.cpp)
target_link_libraries(compare scenario_id scenario_base scenario_ref ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_executable(generate_scenario  scenario_id/src/cuda_utils.cu main/src/generate_scenario.cpp main/src/generate_scenario.cu main/src/util/SimpleArgumentParser.cpp)
set_property(TARGET generate_scenario PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_property(TARGET generate_scenario PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(generate_scenario ${CUDA_LIBRARIES})