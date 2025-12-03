#ifndef __UT_ROBOT_A2_SPORT_CLIENT_HPP__
#define __UT_ROBOT_A2_SPORT_CLIENT_HPP__

#include <limits>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>
#include "sport_api.hpp"

namespace unitree
{
  namespace robot
  {
    namespace a2
    {
      const int ID_PASSIVE = 0;
      const int ID_STAND_DOWN = 1;
      const int ID_STAND_UP = 2;
      const int ID_DEFAULT_MODE = 3;
      const int ID_RUNNING_MODE = 4;
      const int ID_CLIMB_MODE = 5;
      const int ID_LEFT_SIDE_GAIT = 6;
      const int ID_RIGHT_SIDE_GAIT = 7;
      const int ID_HANDSTAND = 8;
      const int ID_BIPED_STAND = 9;
      const int ID_FRONT_FLIP = 10;
      const int ID_BACK_FLIP = 11;
      const int ID_RECOVERY = 12;
      const int ID_BASE_HEIGHT_CTRL = 13;

      class SportClient : public Client
      {
      public:
        SportClient() : Client(ROBOT_SPORT_SERVICE_NAME, false) {}
        ~SportClient() {}

        /*Init*/
        void Init()
        {
          SetApiVersion(ROBOT_SPORT_API_VERSION);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_DAMP);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_BALANCESTAND);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_STOPMOVE);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_STANDUP);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_STANDDOWN);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_RECOVERYSTAND);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_EULER);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_MOVE);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_SWITCHGAIT);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_BODYHEIGHT);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_SPEEDLEVEL);
          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_SETAUTORECOVERY);

          UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_SPORT_API_ID_GETSTATE);
        }

        /*High Level API Call*/
        int32_t Damp()
        {
          std::string parameter, data;
          return Call(ROBOT_SPORT_API_ID_DAMP, parameter, data);
        }

        int32_t BalanceStand()
        {
          std::string parameter, data;
          return Call(ROBOT_SPORT_API_ID_BALANCESTAND, parameter, data);
        }
        int32_t StopMove()
        {
          std::string parameter, data;
          return Call(ROBOT_SPORT_API_ID_STOPMOVE, parameter, data);
        }

        int32_t StandUp()
        {
          std::string parameter, data;
          return Call(ROBOT_SPORT_API_ID_STANDUP, parameter, data);
        }

        int32_t StandDown()
        {
          std::string parameter, data;
          return Call(ROBOT_SPORT_API_ID_STANDDOWN, parameter, data);
        }

        int32_t RecoveryStand()
        {
          std::string parameter, data;
          return Call(ROBOT_SPORT_API_ID_RECOVERYSTAND, parameter, data);
        }

        int32_t Euler(float roll, float pitch, float yaw)
        {
          std::string parameter, data;
          go2::JsonizeVec3 json;
          json.x = roll;
          json.y = pitch;
          json.z = yaw;
          parameter = common::ToJsonString(json);
          return Call(ROBOT_SPORT_API_ID_EULER, parameter, data);
        }

        int32_t Move(float vx, float vy, float vyaw)
        {
          std::string parameter, data;
          go2::JsonizeVec3 json;
          json.x = vx;
          json.y = vy;
          json.z = vyaw;
          parameter = common::ToJsonString(json);
          return Call(ROBOT_SPORT_API_ID_MOVE, parameter, data);
        }

        int32_t SwitchGait(int gait_type)
        {
          std::string parameter, data;
          go2::JsonizeDataInt json;
          json.data = gait_type;
          parameter = common::ToJsonString(json);
          return Call(ROBOT_SPORT_API_ID_SWITCHGAIT, parameter, data);
        }

        int32_t BodyHeight(float height)
        {
          std::string parameter, data;
          go2::JsonizeDataFloat json;
          json.data = height;
          parameter = common::ToJsonString(json);
          return Call(ROBOT_SPORT_API_ID_BODYHEIGHT, parameter, data);
        }

        int32_t SpeedLevel(int level)
        {
          std::string parameter, data;
          go2::JsonizeDataInt json;
          json.data = level;
          parameter = common::ToJsonString(json);
          return Call(ROBOT_SPORT_API_ID_SPEEDLEVEL, parameter, data);
        }

        int32_t SetAutoRecovery(int switch_on)
        {
          std::string parameter, data;
          go2::JsonizeDataInt json;
          json.data = switch_on;
          parameter = common::ToJsonString(json);
          return Call(ROBOT_SPORT_API_ID_SETAUTORECOVERY, parameter, data);
        }

        int32_t GetState(std::map<std::string, std::string> &state_map)
        {
          std::string parameter, data;
          int32_t ret = Call(ROBOT_SPORT_API_ID_GETSTATE, parameter, data);
          common::FromJsonString(data, state_map);
          return ret;
        }
      };
    } // namespace a2

  } // namespace robot
} // namespace unitree
#endif // __UT_ROBOT_A2_LOCO_CLIENT_HPP__
