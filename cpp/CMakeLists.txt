project(ETH3DCamera)
cmake_minimum_required(VERSION 2.8)

# Settings.
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not specified, using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_definitions("-O2 -msse2 -msse3 -std=c++11")
include_directories(
    ./src
)

# Executable.
add_executable(ETH3DCamera
    src/camera.cc
    src/camera.h
    src/main.cc
)
