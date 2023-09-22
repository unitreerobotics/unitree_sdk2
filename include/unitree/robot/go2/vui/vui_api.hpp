#ifndef __UT_ROBOT_GO2_VUI_API_HPP__
#define __UT_ROBOT_GO2_VUI_API_HPP__

#include <unitree/common/json/jsonize.hpp>

namespace unitree
{
namespace robot
{
namespace go2
{
/*service name*/
const std::string ROBOT_VUI_SERVICE_NAME = "vui";

/*api version*/
const std::string ROBOT_VUI_API_VERSION = "1.0.0.0";

/*api id*/
const int32_t ROBOT_VUI_API_ID_SETBRIGHTNESS       = 1005;
const int32_t ROBOT_VUI_API_ID_GETBRIGHTNESS       = 1006;
}
}
}

#endif //__UT_ROBOT_GO2_VUI_API_HPP__
