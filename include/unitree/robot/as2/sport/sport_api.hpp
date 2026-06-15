#ifndef __UT_ROBOT_AS2_SPORT_API_HPP__
#define __UT_ROBOT_AS2_SPORT_API_HPP__

#include <unitree/common/json/jsonize.hpp>
namespace unitree
{
    namespace robot
    {
        namespace as2
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
            const int32_t ROBOT_SPORT_API_ID_BODYPOSITION = 1009;
            const int32_t ROBOT_SPORT_API_ID_SWITCHGAIT = 1011;
            const int32_t ROBOT_SPORT_API_ID_BODYHEIGHT = 1013;
            const int32_t ROBOT_SPORT_API_ID_SPEEDLEVEL = 1015;
           
            const int32_t ROBOT_SPORT_API_ID_LEFTSIDEGAIT = 1016;
            const int32_t ROBOT_SPORT_API_ID_RIGHTSIDEGAIT = 1017;
            const int32_t ROBOT_SPORT_API_ID_HANDSTAND = 1018;
            const int32_t ROBOT_SPORT_API_ID_BIPEDSTAND = 1019;
            const int32_t ROBOT_SPORT_API_ID_FRONTFLIP = 1020;
            const int32_t ROBOT_SPORT_API_ID_BACKFLIP = 1021;
            const int32_t ROBOT_SPORT_API_ID_GREETING = 1022;
            const int32_t ROBOT_SPORT_API_ID_HEART = 1023;
            const int32_t ROBOT_SPORT_API_ID_CONTENT = 1024;
            const int32_t ROBOT_SPORT_API_ID_DANCE1 = 1025;
            const int32_t ROBOT_SPORT_API_ID_DANCE2 = 1026;
            const int32_t ROBOT_SPORT_API_ID_HANDSHAKE = 1027;
            const int32_t ROBOT_SPORT_API_ID_STRETCH = 1028;
            const int32_t ROBOT_SPORT_API_ID_SIT = 1029;
            const int32_t ROBOT_SPORT_API_ID_FRONTJUMP = 1030;
            const int32_t ROBOT_SPORT_API_ID_PUSHUP = 1031;
            const int32_t ROBOT_SPORT_API_ID_UPJUMP = 1032;

            const int32_t ROBOT_SPORT_API_ID_SETAUTORECOVERY = 1040;
            const int32_t ROBOT_SPORT_API_ID_SWITCHJOYSTICK = 1041;

            // Get state
            const int32_t ROBOT_SPORT_API_ID_GETSTATE = 1034;

        }

    }
}

#endif //__UT_ROBOT_As2_SPORT_API_HPP__
