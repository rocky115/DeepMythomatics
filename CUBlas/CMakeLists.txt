cmake_minimum_required(VERSION 3.13)
project(CUBlas)

set(CMAKE_CXX_STANDARD 14)


set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# Find the CUDA package
find_package(CUDA REQUIRED)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS} include)

# Set the source files
set(SOURCES
        src/main.cu
        src/RMD-batch-svd.cu
        src/RMD-batch-qr.cu
        )

# Set the target executable
cuda_add_executable(${PROJECT_NAME} ${SOURCES})

# Link against the CUDA and cuBLAS libraries
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} cublas)

# Set the output directory for the executable
set_target_properties(${PROJECT_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

# Specify the C++ standard
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 11)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

