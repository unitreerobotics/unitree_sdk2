#ifndef __UT_ROBOT_H1_LOCO_H1_LOCO_API_
#define __UT_ROBOT_H1_LOCO_H1_LOCO_API_

#include <unitree/common/json/jsonize.hpp>

using namespace unitree::common;

namespace unitree {
namespace robot {
namespace h1 {
/*service name*/
const std::string LOCO_SERVICE_NAME = "loco";

/*api version*/
const std::string LOCO_API_VERSION = "1.0.0.1";

/*api id*/
const int32_t ROBOT_API_ID_LOCO_MOVE = 2001;
const int32_t ROBOT_API_ID_LOCO_STOP = 2002;
const int32_t ROBOT_API_ID_LOCO_START = 2003;
const int32_t ROBOT_API_ID_LOCO_STAND_SWITCH = 2004;
const int32_t ROBOT_API_ID_LOCO_INCREASE_SWING_HEIGHT = 2005;
const int32_t ROBOT_API_ID_LOCO_DECREASE_SWING_HEIGHT = 2006;
const int32_t ROBOT_API_ID_LOCO_ARM_CTRL = 2007;

class MoveParameter : public Jsonize {
 public:
  MoveParameter() : vx(0.0), vy(0.0), vyaw(0.0) {}

  ~MoveParameter() {}

 public:
  void fromJson(JsonMap& json) {
    FromJson(json["vx"], vx);
    FromJson(json["vy"], vy);
    FromJson(json["vyaw"], vyaw);
  }

  void toJson(JsonMap& json) const {
    ToJson(vx, json["vx"]);
    ToJson(vy, json["vy"]);
    ToJson(vyaw, json["vyaw"]);
  }

 public:
  float vx;
  float vy;
  float vyaw;
};

class ArmCtrlParameter : public Jsonize {
 public:
  ArmCtrlParameter() {}

  ~ArmCtrlParameter() {}

  void fromJson(JsonMap& json) {
    FromJson(json["q"], q);
    FromJson(json["dq"], dq);
    FromJson(json["kp"], kp);
    FromJson(json["kd"], kd);
    FromJson(json["tau_ff"], tau_ff);
    FromJson(json["weight"], weight);
  }

  void toJson(JsonMap& json) const {
    ToJson(q, json["q"]);
    ToJson(dq, json["dq"]);
    ToJson(kp, json["kp"]);
    ToJson(kd, json["kd"]);
    ToJson(tau_ff, json["tau_ff"]);
    ToJson(weight, json["weight"]);
  }

  std::vector<float> q;
  std::vector<float> dq;
  std::vector<float> kp;
  std::vector<float> kd;
  std::vector<float> tau_ff;

  float weight;
};

}  // namespace h1
}  // namespace robot
}  // namespace unitree

#endif  // __UT_ROBOT_H1_LOCO_H1_LOCO_API_
