#pragma once

#include <unitree/common/decl.hpp>

namespace unitree {
namespace robot {
    namespace r1 {
        UT_DECL_ERR( UT_ROBOT_LOCO_ERR_LOCOSTATE_NOT_AVAILABLE, 7301, "LocoState not available." )
        UT_DECL_ERR( UT_ROBOT_LOCO_ERR_SCHEDULER_STATE_NOT_AVAILABLE, 7301, "LocoState not available." )
        UT_DECL_ERR( UT_ROBOT_LOCO_ERR_INVALID_FSM_ID, 7302, "Invalid fsm id." )
        UT_DECL_ERR( UT_ROBOT_LOCO_ERR_INVALID_TASK_ID, 7303, "Invalid task id." )
        UT_DECL_ERR( UT_ROBOT_LOCO_ERROR_FSM_ID_RETURN_DENIED, 7304, "FSM ID return denied." )
    }  // namespace r1
}  // namespace robot
}  // namespace unitree
