#pragma once

#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>
#include "g1_arm_action_api.hpp"

namespace unitree {
namespace robot {
namespace g1 {

  /**
   * @brief Arm action client
   * 
   * The arm action server provides some upper body actions.
   * The controller is based on the `rt/arm_sdk` interface.
   * 
   * The current state of the arm will be published on the `rt/arm/action/state` topic:
   * {
   *   "holding": false,  # Whether to hold the position after the action ends; will release after a maximum of 20 seconds
   *   "id": 99,          # Current action ID
   *   "name": "release_arm" # Current action name
   * }
   */
class G1ArmActionClient : public Client {
  public:
    G1ArmActionClient() : Client(ARM_ACTION_SERVICE_NAME, false) {}
    ~G1ArmActionClient() {}
  
    /*Init*/
    void Init() {
      SetApiVersion(ARM_ACTION_API_VERSION);
      UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_ARM_ACTION_EXECUTE_ACTION);
      UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_ARM_ACTION_GET_ACTION_LIST);
      UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_ARM_ACTION_EXECUTE_CUSTOM_ACTION);
      UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_ARM_ACTION_STOP_CUSTOM_ACTION);
  }
  
  /*API Call*/
  /**
   * @brief Execute Unitree's internal preset actions
   * Indexed by ID, where 99 is to release the action.
   * Some actions will hold the position at the last keyframe after completion; 
   * you can send id = 99 or the same id to release.
   * 
   * Attention:
   *   Some actions will not be displayed on the APP, 
   *   but can be executed by the program.
   *   These actions may cause the robot to fall,
   *   so please execute them with caution.
   */
  int32_t ExecuteAction(int32_t action_id) {
    std::string parameter, data;
    parameter = R"({"action_id":)" + std::to_string(action_id) + R"(})";
    return Call(ROBOT_API_ID_ARM_ACTION_EXECUTE_ACTION, parameter, data);
  }

  /**
   * @brief Execute custom teach actions, indexed by name
   * 
   * # rt/arm/action/state
   * {
   *   "holding": false,  # teach actions do not include holding
   *   "id": 100,         # Always 100, the action is determined by the name
   *   "name": "sth"      # Current teach action name
   * }
   */
  int32_t ExecuteAction(const std::string &action_name) {
    std::string parameter, data;
    parameter = R"({"action_name":")" + action_name + R"("})";
    return Call(ROBOT_API_ID_ARM_ACTION_EXECUTE_CUSTOM_ACTION, parameter, data);
  }

  /* Stop Custom Action */
  int32_t StopCustomAction() {
    std::string parameter, data;
    return Call(ROBOT_API_ID_ARM_ACTION_STOP_CUSTOM_ACTION, parameter, data);
  }

  int32_t GetActionList(std::string &data) {
    std::string parameter;
    return Call(ROBOT_API_ID_ARM_ACTION_GET_ACTION_LIST, parameter, data);
  }

  /*Action List*/
  std::map<std::string, int32_t> action_map = {
    {"release arm", 99},
    {"two-hand kiss", 11},
    {"left kiss", 12},
    {"right kiss", 12},
    {"hands up", 15},
    {"clap", 17},
    {"high five", 18},
    {"hug", 19},
    {"heart", 20},
    {"right heart", 21},
    {"reject", 22},
    {"right hand up", 23},
    {"x-ray", 24},
    {"face wave", 25},
    {"high wave", 26},
    {"shake hand", 27},
  };
};


} // namespace g1
} // namespace robot
} // namespace unitree 