#ifndef __UT_ROBOT_B2_SPORT_API_HPP__
#define __UT_ROBOT_B2_SPORT_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
/*service name*/
const std::string ROBOT_SPORT_SERVICE_NAME = "sport";

/*api version*/
const std::string ROBOT_SPORT_API_VERSION = "1.0.0.1";

/*api id*/
const int32_t ROBOT_SPORT_API_ID_DAMP               = 1001;
const int32_t ROBOT_SPORT_API_ID_BALANCESTAND       = 1002;
const int32_t ROBOT_SPORT_API_ID_STOPMOVE           = 1003;
const int32_t ROBOT_SPORT_API_ID_STANDUP            = 1004;
const int32_t ROBOT_SPORT_API_ID_STANDDOWN          = 1005;
const int32_t ROBOT_SPORT_API_ID_RECOVERYSTAND      = 1006;
const int32_t ROBOT_SPORT_API_ID_EULER              = 1007;
const int32_t ROBOT_SPORT_API_ID_MOVE               = 1008;
const int32_t ROBOT_SPORT_API_ID_SIT                = 1009;
const int32_t ROBOT_SPORT_API_ID_SWITCHGAIT         = 1010;
const int32_t ROBOT_SPORT_API_ID_BODYHEIGHT         = 1011;
const int32_t ROBOT_SPORT_API_ID_FOOTRAISEHEIGHT    = 1012;
const int32_t ROBOT_SPORT_API_ID_SPEEDLEVEL         = 1013;
const int32_t ROBOT_SPORT_API_ID_TRAJECTORYFOLLOW   = 1014;
const int32_t ROBOT_SPORT_API_ID_CONTINUOUSGAIT     = 1015;
const int32_t ROBOT_SPORT_API_ID_FRONTJUMP          = 1016;
const int32_t ROBOT_SPORT_API_ID_ECONOMICGAIT       = 1017;
const int32_t ROBOT_SPORT_API_ID_MOVETOPOS          = 1018;

}
}
}

#endif //__UT_ROBOT_B2_SPORT_API_HPP__
