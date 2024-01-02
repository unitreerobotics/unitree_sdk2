#ifndef __UT_ROBOT_B2_SPORT_CLIENT_HPP__
#define __UT_ROBOT_B2_SPORT_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
/*
 * PathPoint
 */
struct stPathPoint
{
  float timeFromStart;
  float x;
  float y;
  float yaw;
  float vx;
  float vy;
  float vyaw;
};

typedef struct stPathPoint PathPoint;

/*
 * SportClient
 */
class SportClient : public Client
{
public:
    explicit SportClient(bool enableLease = false);
    ~SportClient();

    void Init();

    /*
     * @brief Damp
     * @api: 1001
     */
    int32_t Damp();

    /*
     * @brief BalanceStand
     * @api: 1002
     */
    int32_t BalanceStand();

    /*
     * @brief StopMove
     * @api: 1003
     */
    int32_t StopMove();

    /*
     * @brief StandUp
     * @api: 1004
     */
    int32_t StandUp();

    /*
     * @brief StandDown
     * @api: 1005
     */
    int32_t StandDown();

    /*
     * @brief RecoveryStand
     * @api: 1006
     */
    int32_t RecoveryStand();

    /*
     * @brief Euler
     * @api: 1007
     */
    int32_t Euler(float roll, float pitch, float yaw);

    /*
     * @brief Move
     * @api: 1008
     */
    int32_t Move(float vx, float vy, float vyaw);

    /*
     * @brief Sit
     * @api: 1009
     */
    int32_t Sit();

    /*
     * @brief SwitchGait
     * @api: 1010
     */
    int32_t SwitchGait(int d);

    /*
     * @brief BodyHeight
     * @api: 1011
     */
    int32_t BodyHeight(float height);

    /*
     * @brief FootRaiseHeight
     * @api: 1012
     */
    int32_t FootRaiseHeight(float height);

    /*
     * @brief SpeedLevel
     * @api: 1013
     */
    int32_t SpeedLevel(int level);

    /*
     * @brief TrajectoryFollow
     * @api: 1014
     */
    int32_t TrajectoryFollow(std::vector<PathPoint>& path);

    /*
     * @brief ContinuousGait
     * @api: 1015
     */
    int32_t ContinuousGait(bool flag);

    /*
     * @brief FrontJump
     * @api: 1016
     */
    int32_t FrontJump();

    /*
     * @brief EconomicGait
     * @api: 1017
     */
    int32_t EconomicGait(bool flag);

    /*
     * @brief MoveToPos
     * @api: 1018
     */
    int32_t MoveToPos(float x, float y, float yaw);

};
}
}
}

#endif//__UT_ROBOT_B2_SPORT_CLIENT_HPP__
