#ifndef __UT_ROBOT_A2_SPORT_API_HPP__
#define __UT_ROBOT_A2_SPORT_API_HPP__

#include <unitree/common/json/jsonize.hpp>
namespace unitree
{
    namespace robot
    {
        namespace a2
        {
            /*service name*/
            const std::string ROBOT_SPORT_SERVICE_NAME = "sport";

            /*api version*/
            const std::string ROBOT_SPORT_API_VERSION = "1.0.0.1";

            /*api id*/
            // Set motion
            const int32_t ROBOT_SPORT_API_ID_DAMP = 1001;
            const int32_t ROBOT_SPORT_API_ID_BALANCESTAND = 1002;
            const int32_t ROBOT_SPORT_API_ID_STOPMOVE = 1003;
            const int32_t ROBOT_SPORT_API_ID_STANDUP = 1004;
            const int32_t ROBOT_SPORT_API_ID_STANDDOWN = 1005;
            const int32_t ROBOT_SPORT_API_ID_RECOVERYSTAND = 1006;
            const int32_t ROBOT_SPORT_API_ID_EULER = 1007;
            const int32_t ROBOT_SPORT_API_ID_MOVE = 1008;
            const int32_t ROBOT_SPORT_API_ID_SWITCHGAIT = 1011;
            const int32_t ROBOT_SPORT_API_ID_BODYHEIGHT = 1013;
            const int32_t ROBOT_SPORT_API_ID_SPEEDLEVEL = 1015;
            const int32_t ROBOT_SPORT_API_ID_SETAUTORECOVERY = 1040;

            // Get state
            const int32_t ROBOT_SPORT_API_ID_GETSTATE = 1034;

        }

    }
}

#endif //__UT_ROBOT_A2_SPORT_API_HPP__
