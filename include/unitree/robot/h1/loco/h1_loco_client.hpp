#ifndef __UT_ROBOT_H1_LOCO_H1_LOCO_CLIENT_
#define __UT_ROBOT_H1_LOCO_H1_LOCO_CLIENT_

#include <unitree/robot/client/client.hpp>
#include "h1_loco_api.hpp"

namespace unitree {
namespace robot {
namespace h1 {
class H1LocoClient : public Client {
 public:
  H1LocoClient() : Client(LOCO_SERVICE_NAME, false) {}
  ~H1LocoClient() {}

  /*Init*/
  void Init() {
    SetApiVersion(LOCO_API_VERSION);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_MOVE);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_START);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_STOP);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_STAND_SWITCH);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_INCREASE_SWING_HEIGHT);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_DECREASE_SWING_HEIGHT);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_ARM_CTRL);
  };

  /*API Call*/
  int32_t Move(float vx, float vy, float vyaw) {
    MoveParameter param;

    param.vx = vx;
    param.vy = vy;
    param.vyaw = vyaw;

    std::string parameter = common::ToJsonString(param);
    std::string data;

    return Call(ROBOT_API_ID_LOCO_MOVE, parameter, data);
  }

  int32_t EnableCtrl() {
    std::string parameter, data;
    return Call(ROBOT_API_ID_LOCO_START, parameter, data);
  }

  int32_t Stop() {
    std::string parameter, data;
    return Call(ROBOT_API_ID_LOCO_STOP, parameter, data);
  }

  int32_t StandSwitch() {
    std::string parameter, data;
    return Call(ROBOT_API_ID_LOCO_STAND_SWITCH, parameter, data);
  }

  int32_t IncreaseSwingHeight() {
    std::string parameter, data;
    return Call(ROBOT_API_ID_LOCO_INCREASE_SWING_HEIGHT, parameter, data);
  }

  int32_t DecreaseSwingHeight() {
    std::string parameter, data;
    return Call(ROBOT_API_ID_LOCO_DECREASE_SWING_HEIGHT, parameter, data);
  }

  int32_t ArmCtrl(std::vector<float> q, std::vector<float> dq, std::vector<float> kp, std::vector<float> kd,
                  std::vector<float> tau_ff, float weight) {
    ArmCtrlParameter param;

    param.q = q;
    param.dq = dq;
    param.kp = kp;
    param.kd = kd;
    param.tau_ff = tau_ff;
    param.weight = weight;

    std::string parameter = common::ToJsonString(param);
    std::string data;

    return Call(ROBOT_API_ID_LOCO_ARM_CTRL, parameter, data);
  }
};
}  // namespace h1

}  // namespace robot
}  // namespace unitree

#endif // __UT_ROBOT_H1_LOCO_H1_LOCO_CLIENT_
