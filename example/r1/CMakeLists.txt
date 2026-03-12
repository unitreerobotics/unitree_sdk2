cmake_minimum_required(VERSION 3.10)

add_executable(r1_loco_client high_level/r1_loco_client_example.cpp)
target_link_libraries(r1_loco_client unitree_sdk2)
