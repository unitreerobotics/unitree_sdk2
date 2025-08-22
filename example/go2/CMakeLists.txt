find_package(OpenCV REQUIRED)

add_executable(go2_video_client_custom go2_video_client_custom.cpp)
target_include_directories(go2_video_client_custom PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_link_libraries(go2_video_client_custom unitree_sdk2 ${OpenCV_LIBS})
target_link_libraries(go2_video_client_custom
    unitree_sdk2
    ${OpenCV_LIBS}
    curl  
)


# add_executable(go2_trajectory_follow go2_trajectory_follow.cpp)
# target_link_libraries(go2_trajectory_follow unitree_sdk2)

# add_executable(go2_sport_client go2_sport_client.cpp)
# target_link_libraries(go2_sport_client unitree_sdk2)

# add_executable(go2_low_level go2_low_level.cpp)
# target_link_libraries(go2_low_level unitree_sdk2)

# add_executable(go2_stand_example go2_stand_example.cpp)
# target_link_libraries(go2_stand_example unitree_sdk2)


# add_executable(go2_robot_state_client go2_robot_state_client.cpp)
# target_link_libraries(go2_robot_state_client unitree_sdk2)

# add_executable(go2_video_client go2_video_client.cpp)
# target_link_libraries(go2_video_client unitree_sdk2)

# add_executable(go2_vui_client go2_vui_client.cpp)
# target_link_libraries(go2_vui_client unitree_sdk2)
