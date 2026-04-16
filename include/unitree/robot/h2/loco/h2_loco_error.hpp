#ifndef __UT_ROBOT_H2_LOCO_ERROR_HPP__
#define __UT_ROBOT_H2_LOCO_ERROR_HPP__

#include <unitree/common/decl.hpp>

namespace unitree {
namespace robot {
namespace h2 {
UT_DECL_ERR(UT_ROBOT_LOCO_ERR_LOCOSTATE_NOT_AVAILABLE, 7301, "LocoState not available.")
UT_DECL_ERR(UT_ROBOT_LOCO_ERR_SCHEDULER_STATE_NOT_AVAILABLE, 7301, "LocoState not available.")
UT_DECL_ERR(UT_ROBOT_LOCO_ERR_INVALID_FSM_ID, 7302, "Invalid fsm id.")
UT_DECL_ERR(UT_ROBOT_LOCO_ERR_INVALID_TASK_ID, 7303, "Invalid task id.")
}  // namespace h2
}  // namespace robot
}  // namespace unitree

#endif // __UT_ROBOT_H2_LOCO_ERROR_HPP__

