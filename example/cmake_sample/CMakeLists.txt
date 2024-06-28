cmake_minimum_required(VERSION 3.5.0)
project(cmake_sample)

## Set compiler to use c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Additional cmake module path
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")

## Find required packages
find_package(unitree_sdk2 REQUIRED)

## Now we can build our executables
add_executable(low_level low_level.cpp)
target_link_libraries(low_level unitree_sdk2)

add_executable(stand_example_go2 stand_example_go2.cpp)
target_link_libraries(stand_example_go2 unitree_sdk2)

add_executable(stand_example_b2  stand_example_b2.cpp)
target_link_libraries(stand_example_b2 unitree_sdk2)