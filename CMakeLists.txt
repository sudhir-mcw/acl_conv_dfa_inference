cmake_minimum_required(VERSION 3.10)
project(inference)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(ZLIB REQUIRED)

# Manually find cnpy
find_library(CNPY_LIBRARY cnpy REQUIRED)
find_path(CNPY_INCLUDE_DIR cnpy.h REQUIRED)

# Set the path to ARM Compute Library, CNPY_LIBRARY and CNPY_INCLUDE_DIR
# Make sure to adjust the path to where ARM Compute Library, CNPY is installed before building
set(ARM_COMPUTE_DIR "/data/ComputeLibrary" CACHE PATH "Path to ARM Compute Library")
set(ARM_COMPUTE_LIBRARY "/data/ComputeLibrary/build/libarm_compute.so" CACHE FILEPATH "Path to ARM Compute Library so")
set(CNPY_LIBRARY "/data/cnpy/build/libcnpy.so" CACHE FILEPATH "Path to cnpy library so")
set(CNPY_INCLUDE_DIR "/usr/include" CACHE PATH "Path to cnpy headers")


# create directories for input, weight and output
make_directory(${CMAKE_SOURCE_DIR}/new_test)
make_directory(${CMAKE_SOURCE_DIR}/new_test/input)
make_directory(${CMAKE_SOURCE_DIR}/new_test/weight)
make_directory(${CMAKE_SOURCE_DIR}/new_test/output)

# Add include directories
include_directories(
    ${ARM_COMPUTE_DIR}/include
    ${ARM_COMPUTE_DIR}
    ${CNPY_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

# Add the executable
add_executable(conv_dfa_wt_pad conv_dfa_wt_pad.cc)

# Link libraries
target_link_libraries(conv_dfa_wt_pad
    PRIVATE
    ${CNPY_LIBRARY}
    ${ARM_COMPUTE_LIBRARY}
    ZLIB::ZLIB
    arm_compute
)