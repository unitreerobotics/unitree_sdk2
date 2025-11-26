#ifndef __UT_ROBOT_G1_AGV_ERROR_HPP__
#define __UT_ROBOT_G1_AGV_ERROR_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace robot
{
namespace g1
{

UT_DECL_ERR(UT_ROBOT_G1_AGV_ERR_NOT_INIT,   9101,   "Module not initialized.")
UT_DECL_ERR(UT_ROBOT_G1_AGV_ERR_EXEC_MOVE,   9102,   "Failed to execute move command.")
UT_DECL_ERR(UT_ROBOT_G1_AGV_ERR_EXEC_HEIGHT_ADJUST,   9103,   "Failed to execute height adjust command.")

}
}
}

#endif // __UT_ROBOT_G1_AGV_ERROR_HPP__
