cmake_minimum_required(VERSION 3.10)

add_executable(h2_loco_client high_level/h2_loco_client_example.cpp)
target_link_libraries(h2_loco_client unitree_sdk2)

add_executable(h2_arm_sdk_dds_example high_level/h2_arm_sdk_dds_example.cpp)
target_link_libraries(h2_arm_sdk_dds_example unitree_sdk2)

add_executable(h2_ankle_swing_example low_level/h2_ankle_swing_example.cpp)
target_link_libraries(h2_ankle_swing_example unitree_sdk2)

find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND AND ${yaml-cpp_VERSION} VERSION_GREATER_EQUAL "0.6")
    add_executable(h2_dual_arm_example low_level/h2_dual_arm_example.cpp)
    target_link_libraries(h2_dual_arm_example PRIVATE unitree_sdk2 yaml-cpp)
    target_compile_definitions(h2_dual_arm_example PUBLIC
        H2_DUAL_ARM_BEHAVIOR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/low_level/behavior_lib/")
endif()

find_package(Boost COMPONENTS program_options)
if(Boost_FOUND)
    add_executable(h2_termination low_level/terminations.cpp)
    target_link_libraries(h2_termination unitree_sdk2 Boost::program_options)

    add_executable(h2_arm_action_example high_level/h2_arm_action_example.cpp)
    target_link_libraries(h2_arm_action_example unitree_sdk2 Boost::program_options)
endif()

add_executable(h2d_agv_client_example h2d/h2d_agv_client_example.cpp)
target_link_libraries(h2d_agv_client_example unitree_sdk2)

add_executable(h2d_arm_example h2d/h2d_arm_example.cpp)
target_link_libraries(h2d_arm_example unitree_sdk2)

add_executable(h2d_height_control h2d/h2d_height_control.cpp)
target_link_libraries(h2d_height_control unitree_sdk2)

add_executable(h2d_waist_example h2d/h2d_waist_example.cpp)
target_link_libraries(h2d_waist_example unitree_sdk2)
