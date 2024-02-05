project(crc_module)
cmake_minimum_required(VERSION 3.5)

find_package(pybind11 REQUIRED PATHS /home/unitree/miniforge3/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 NO_DEFAULT_PATH)

add_library(crc_module MODULE crc_pybind.cpp)
include_directories(/usr/local/include/ddscxx /usr/local/include/iceoryx/v2.0.2)
link_libraries(unitree_sdk2 ddsc ddscxx rt pthread)


# Link pybind11 to your module
target_link_libraries(crc_module PRIVATE pybind11::module)

# Disable symbol visibility for other targets
set_target_properties(crc_module PROPERTIES CXX_VISIBILITY_PRESET "hidden")

# Optional: set the output directory for the built module
set_target_properties(crc_module PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(crc_module PROPERTIES PREFIX "")

# Set C++ standard, pybind11 requires C++14 or later
set_property(TARGET crc_module PROPERTY CXX_STANDARD 14)