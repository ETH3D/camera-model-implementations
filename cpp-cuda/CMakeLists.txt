project(ETH3DCameraCUDA)
cmake_minimum_required(VERSION 2.8)

# External dependency: CUDA
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11")

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
cuda_add_executable(ETH3DCameraCUDA
    src/camera.cc
    src/camera.h
    src/cuda-util.h
    src/main.cc
    src/main.cu
    src/main.cuh
)
