add_executable(as2_sport_client  sport/as2_sport_client.cpp)
target_link_libraries(as2_sport_client unitree_sdk2)

add_executable(as2_sport_state  sport/as2_sport_state.cpp)
target_link_libraries(as2_sport_state unitree_sdk2)
