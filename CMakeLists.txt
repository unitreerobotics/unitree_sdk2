project(unitree_sdk2)
cmake_minimum_required(VERSION 3.5)

SET(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES unitree_sdk2
    CATKIN_DEPENDS
)

include_directories(
    include
    ${catkin_INCLUDE_DIRS}
)

# include_directories(/usr/local/include/ddscxx /usr/local/include/iceoryx/v2.0.2)
# link_libraries(unitree_sdk2 ddsc ddscxx rt pthread)
if(ENABLE_BUILD)
    add_executable(test_publisher example/helloworld/publisher.cpp example/helloworld/HelloWorldData.cpp)
    add_executable(test_subscriber example/helloworld/subscriber.cpp example/helloworld/HelloWorldData.cpp)
    add_executable(high_follow_sin example/high_level/follow_sin.cpp)
    add_executable(sportmode_test example/high_level/sportmode_test.cpp)
    add_executable(low_level example/low_level/low_level.cpp)
    add_executable(stand_example_go2 example/low_level/stand_example_go2.cpp)
    add_executable(stand_example_b2 example/low_level/stand_example_b2.cpp)
    add_executable(wireless example/wireless/wireless.cpp)
    add_executable(advanced_gamepad example/advanced_gamepad/main.cpp)
    add_executable(state_machine_example example/state_machine/main.cpp)
    add_executable(test_jsonize example/jsonize/test_jsonize.cpp)
    add_executable(robot_state_client_example example/client/robot_state_client_example.cpp)
    add_executable(sport_client_example example/client/sport_client_example.cpp)
    add_executable(video_client_example example/client/video_client_example.cpp)
    add_executable(vui_client_example example/client/vui_client_example.cpp)
    add_executable(h1_arm_sdk_dds_example example/humanoid/high_level/h1_arm_sdk_dds_example.cpp)
    add_executable(h1_loco_example_client example/humanoid/high_level/h1_loco_example_client.cpp)
    add_executable(h1_low_level_example example/humanoid/low_level/humanoid.cpp)
    add_executable(h1_hand_example example/humanoid/low_level/inspire_hand.cpp)
endif()

install(DIRECTORY include/${PROJECT_NAME}/
    DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
