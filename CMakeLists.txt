cmake_minimum_required(VERSION 3.20)

project(NeuralNetwork
        VERSION 3.0.0
        DESCRIPTION "A feedforward neural network written from scratch in C++"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif ()

add_library(neuralnetwork_core STATIC
        src/Neuron.cpp
        src/Layer.cpp
        src/NeuralNetwork.cpp
        src/Utilities.cpp

        include/Neuron.h
        include/Layer.h
        include/NeuralNetwork.h
        include/Utilities.h)

target_include_directories(neuralnetwork_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

find_package(Threads REQUIRED)
target_link_libraries(neuralnetwork_core PUBLIC Threads::Threads)

find_package(TBB QUIET)
if (TBB_FOUND)
    target_link_libraries(neuralnetwork_core PUBLIC TBB::tbb)
    message(STATUS "TBB found; linking <execution> against TBB::tbb")
else ()
    message(STATUS "TBB not found; <execution> policies fall back to serial execution")
endif ()

add_executable(NeuralNetwork
        main.cpp

        datasets/DataTransformations.h
        datasets/car/car.h
        datasets/mnist/mnist.h)

target_include_directories(NeuralNetwork PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/datasets")
target_link_libraries(NeuralNetwork PRIVATE neuralnetwork_core)

foreach (target neuralnetwork_core NeuralNetwork)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive-)
    else ()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif ()
endforeach ()

option(NEURALNETWORK_BUILD_TESTS "Build the NeuralNetwork test suite" ${PROJECT_IS_TOP_LEVEL})

if (NEURALNETWORK_BUILD_TESTS)
    enable_testing()

    add_executable(gradient_check tests/gradient_check.cpp)
    target_link_libraries(gradient_check PRIVATE neuralnetwork_core)

    if (MSVC)
        target_compile_options(gradient_check PRIVATE /W4 /permissive-)
    else ()
        target_compile_options(gradient_check PRIVATE -Wall -Wextra -Wpedantic)
    endif ()

    add_test(NAME gradient_check COMMAND gradient_check)
endif ()
