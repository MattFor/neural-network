cmake_minimum_required(VERSION 3.30)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 26)

add_executable(NeuralNetwork main.cpp
        src/Neuron.cpp
        include/Neuron.h
        src/Layer.cpp
        include/Layer.h
        src/NeuralNetwork.cpp
        include/NeuralNetwork.h
        transformations/zoo.cpp
        transformations/transformations.h
        src/utilities.cpp
        include/Utilities.h)
