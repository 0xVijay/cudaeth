cmake_minimum_required(VERSION 3.18)
project(BruteForceMnemonic CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/config)
include_directories(${CMAKE_SOURCE_DIR}/Tools)

# CUDA compilation flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-arch=sm_60)

# Source files
set(CUDA_SOURCES
    BruteForceMnemonic/Dispatcher.cu
    BruteForceMnemonic/GPU.cu
    BruteForceMnemonic/Helper.cu
    BruteForceMnemonic/KernelStride.cu
    BruteForceMnemonic/kernel.cu
)

set(CXX_SOURCES
    config/Config.cpp
    Tools/tools.cpp
    Tools/utils.cpp
    Tools/segwit_addr.cpp
)

# Create executable
cuda_add_executable(BruteForceMnemonic ${CUDA_SOURCES} ${CXX_SOURCES})

# Link libraries
target_link_libraries(BruteForceMnemonic ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})

# Copy config file to build directory
configure_file(${CMAKE_SOURCE_DIR}/BruteForceMnemonic/config.cfg ${CMAKE_BINARY_DIR}/config.cfg COPYONLY)

# Create tables directory
add_custom_command(TARGET BruteForceMnemonic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/tables_ethereum
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/tables_ethereum/A0.csv ${CMAKE_BINARY_DIR}/tables_ethereum/
)
