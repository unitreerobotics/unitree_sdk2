#pragma once

#include <unitree/common/json/jsonize.hpp>
#include <variant>

namespace unitree {
namespace robot {
namespace g1 {
/*service name*/
const std::string ARM_ACTION_SERVICE_NAME = "arm";

/*api version*/
const std::string ARM_ACTION_API_VERSION = "1.0.0.14";

/*api id*/
const int32_t ROBOT_API_ID_ARM_ACTION_EXECUTE_ACTION = 7106;
const int32_t ROBOT_API_ID_ARM_ACTION_GET_ACTION_LIST = 7107;
const int32_t ROBOT_API_ID_ARM_ACTION_EXECUTE_CUSTOM_ACTION = 7108;
const int32_t ROBOT_API_ID_ARM_ACTION_STOP_CUSTOM_ACTION = 7113;

}  // namespace g1
}  // namespace robot
}  // namespace unitree