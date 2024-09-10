add_executable(h1_loco_client high_level/h1_loco_client_example.cpp)
target_link_libraries(h1_loco_client unitree_sdk2)

add_executable(g1_loco_client high_level/g1_loco_client_example.cpp)
target_link_libraries(g1_loco_client unitree_sdk2)

add_executable(g1_arm5_sdk_dds_example high_level/g1_arm5_sdk_dds_example.cpp)
target_link_libraries(g1_arm5_sdk_dds_example unitree_sdk2)

add_executable(g1_arm7_sdk_dds_example high_level/g1_arm7_sdk_dds_example.cpp)
target_link_libraries(g1_arm7_sdk_dds_example unitree_sdk2)

add_executable(h1_low_level_example low_level/humanoid.cpp)
target_link_libraries(h1_low_level_example unitree_sdk2)

add_executable(h1_27dof_example low_level/h1_27dof_example.cpp)
target_link_libraries(h1_27dof_example unitree_sdk2)

add_executable(g1_23dof_example low_level/g1_23dof_example.cpp)
target_link_libraries(g1_23dof_example unitree_sdk2)

add_executable(g1_29dof_example low_level/g1_29dof_example.cpp)
target_link_libraries(g1_29dof_example unitree_sdk2)

find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    add_executable(g1_dual_arm_example low_level/g1_dual_arm_example.cpp)
    target_include_directories(g1_dual_arm_example PRIVATE ${YAML_CPP_INCLUDE_DIRS})
    target_link_libraries(g1_dual_arm_example PRIVATE unitree_sdk2 ${YAML_CPP_LIBRARIES})
    target_compile_definitions(g1_dual_arm_example PUBLIC BLIB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/low_level/behavior_lib/")
endif()
