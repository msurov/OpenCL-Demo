cmake_minimum_required(VERSION 3.2)
add_custom_target(kernels)
# add_custom_command(TARGET kernels PRE_BUILD COMMAND "python3" ARGS "compose_header.py")
add_custom_command(
    TARGET kernels
    PRE_BUILD COMMAND "python" ARGS "compose_header.py"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
