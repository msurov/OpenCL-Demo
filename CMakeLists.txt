cmake_minimum_required(VERSION 3.7)
project(opencl-demo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCL REQUIRED)
add_subdirectory(kernels)
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
add_dependencies(${PROJECT_NAME} kernels)

# file(READ kernel.cl KERNEL)
# string(MAKE_C_IDENTIFIER "${KERNEL}" FORMATTED)
# message(KERNEL ${FORMATTED})
# 1. File as binary
# 2. 
