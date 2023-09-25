/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/


#include <cmath>

#include <unitree/robot/go2/sport/sport_client.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

#define TOPIC_HIGHSTATE "rt/sportmodestate"

using namespace unitree::common;

enum test_mode
{
  /*---基础运动控制---*/
  normal_stand,      // 0 正常站立
  balance_stand,     // 1 平衡站立
  velocity_move,     // 2 速度控制
  position_control,  // 3 点到点的位置移动
  trajectory_follow, // 4 轨迹跟踪控制
  stand_down,        // 5 趴下
  stand_up,          // 6 站起
  damp,              // 7 软急停
  recovery_stand,    // 8 恢复站立
  /*---特殊动作 ---*/
  sit,           // 9 坐下
  rise_sit,      // 10 从坐下中恢复
  stretch,       // 11 伸懒腰
  wallow,        // 12 打滚
  content,       // 13 开心
  pose,          // 14 摆姿势
  scrape,        // 15 拜年作揖
  front_flip,    // 16 前空翻
  front_jump,    // 17 前跳
  front_pounce,  // 18 向前扑人
  stop_move = 99 // 停止运动
};

const int TEST_MODE = position_control;

class Custom
{
public:
  Custom()
  {
    sport_client.SetTimeout(10.0f);
    sport_client.Init();

    suber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
    //unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_> suber(TOPIC_HIGHSTATE);
    suber->InitChannel(std::bind(&Custom::HighStateHandler, this, std::placeholders::_1), 1);
  };

  // 高层控制实例
  void RobotControl()
  {
    if (get_init_pos)
    {
      px0 = state.position()[0];
      py0 = state.position()[1];
      yaw0 = state.imu_state().rpy()[2];
      get_init_pos = 0;
      std::cout << "init_pos: x0: " << px0 << ", y0: " << py0 << ", yaw0: " << yaw0 << std::endl;
    }

    ct += dt;
    double px_local, py_local, yaw_local;
    double vx_local, vy_local, vyaw_local;
    double px_err, py_err, yaw_err;
    double time_seg, time_temp;

    unitree::robot::go2::PathPoint path_point_tmp;
    std::vector<unitree::robot::go2::PathPoint> path;

    switch (TEST_MODE)
    {
    case normal_stand:            // 0. idle, default stand
      sport_client.SwitchGait(0); // 0:idle; 1:tort; 2:tort running; 3:climb stair; 4:tort obstacle
      sport_client.StandUp();
      break;

    case balance_stand:                  // 1. Balance stand (controlled by dBodyHeight + rpy)
      sport_client.Euler(0.1, 0.2, 0.3); // roll, pitch, yaw
      sport_client.BodyHeight(0.0);      // relative height [-0.18~0.03]
      sport_client.BalanceStand();
      break;

    case velocity_move:                   // 2. target velocity walking (controlled by velocity + yawSpeed)
      sport_client.SwitchGait(1);         // tort
      sport_client.FootRaiseHeight(0.08); //[-0.06~0.03]
      sport_client.SpeedLevel(1);         // 0:slow; 1:normal; 2:fast
      sport_client.Move(0.3, 0, 0.3);
      break;

    case position_control: // 3. target position walking (controlled by position + rpy[0])
      sport_client.SpeedLevel(1);
      sport_client.SwitchGait(1);

      px_local = 0.2;
      py_local = -0.2;
      yaw_local = 1.8;

      px_err = state.position()[0] - (px_local * cos(yaw0) - py_local * sin(yaw0) + px0);
      py_err = state.position()[1] - (px_local * sin(yaw0) + py_local * cos(yaw0) + py0);
      yaw_err = state.imu_state().rpy()[2] - (yaw_local + yaw0);

      sport_client.Move(-1.5 * (px_err * cos(state.imu_state().rpy()[2] - yaw0) + py_err * sin(state.imu_state().rpy()[2] - yaw0)),
                        -1.5 * (-px_err * sin(state.imu_state().rpy()[2] - yaw0) + py_err * cos(state.imu_state().rpy()[2] - yaw0)),
                        -1 * yaw_err);
      break;

    case trajectory_follow: // 4. path mode walking
      sport_client.SwitchGait(1);
      time_seg = 0.2;
      time_temp = ct - time_seg;
      for (int i = 0; i < 30; i++)
      {
        time_temp += time_seg;

        px_local = 0.5 * sin(0.5 * time_temp);
        py_local = 0;
        yaw_local = 0;
        vx_local = 0.5 * cos(0.5 * time_temp);
        vy_local = 0;
        vyaw_local = 0;

        path_point_tmp.timeFromStart = i * time_seg;
        path_point_tmp.x = px_local * cos(yaw0) - py_local * sin(yaw0) + px0;
        path_point_tmp.y = px_local * sin(yaw0) + py_local * cos(yaw0) + py0;
        path_point_tmp.yaw = yaw_local + yaw0;
        path_point_tmp.vx = vx_local * cos(yaw0) - vy_local * sin(yaw0);
        path_point_tmp.vy = vx_local * sin(yaw0) + vy_local * cos(yaw0);
        path_point_tmp.vyaw = vyaw_local;
        path.push_back(path_point_tmp);
      }
      sport_client.TrajectoryFollow(path);
      break;

    case stand_down: // 5. position stand down.
      sport_client.StandDown();
      break;

    case stand_up: // 6. position stand up
      // sport_client.BodyHeight(-0.05);
      sport_client.StandUp();
      break;

    case damp: // 7. damping mode
      sport_client.Damp();
      break;

    case recovery_stand: // 8. recovery stand
      sport_client.RecoveryStand();
      break;

    case sit:
      if (flag == 0)
      {
        sport_client.Sit();
        flag = 1;
      }
      break;

    case rise_sit:
      if (flag == 0)
      {
        sport_client.RiseSit();
        flag = 1;
      }
      break;

    case stretch:
      if (flag == 0)
      {
        sport_client.Stretch();
        flag = 1;
      }
      break;

    case wallow:
      if (flag == 0)
      {
        sport_client.Wallow();
        flag = 1;
      }
      break;

    case content:
      if (flag == 0)
      {
        sport_client.Content();
        flag = 1;
      }
      break;

    case pose:
      if (flag == 0)
      {
        sport_client.Pose(true);
        flag = 1;
      }
      break;

    case scrape:
      if (flag == 0)
      {
        sport_client.Scrape();
        flag = 1;
      }
      break;

    case front_flip:
      if (flag == 0)
      {
        sport_client.FrontFlip();
        flag = 1;
      }
      break;

    case front_jump:
      if (flag == 0)
      {
        sport_client.FrontJump();
        flag = 1;
      }
      break;
    case front_pounce:
      if (flag == 0)
      {
        sport_client.FrontPounce();
        flag = 1;
      }
      break;

    case stop_move: // stop move
      sport_client.StopMove();
      break;

    default:
      sport_client.StopMove();
    }
  };

  // 状态获取函数
  void StateRecv()
  {
    // 输出位置和IMU数据
    std::cout << "Position: " << state.position()[0] << ", " << state.position()[1] << ", " << state.position()[2] << std::endl;
    std::cout << "IMU rpy: " << state.imu_state().rpy()[0] << ", " << state.imu_state().rpy()[1] << ", " << state.imu_state().rpy()[2] << std::endl;
  };

  // 高层数据接收的回调函数
  void HighStateHandler(const void *message)
  {
    state = *(unitree_go::msg::dds_::SportModeState_ *)message;
  };


  unitree_go::msg::dds_::SportModeState_ state;
  unitree::robot::go2::SportClient sport_client;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> suber;

  int get_init_pos = true; // 初始状态获取标志
  double px0, py0, yaw0;   // 初始状态的位置和偏航
  double ct = 0;           // 运行时间
  int flag = 0;            // 特殊动作执行标志
  float dt = 0.01;         // 控制步长0.001~0.01
};

int main()
{
  std::string networkInterface = "enp2s0";
  unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);

  Custom custom;

  double ct=-1;

  while (1)
  { 
    ct+=custom.dt;
    custom.StateRecv();
    if(ct>0)
    {
      custom.RobotControl();
    }
    usleep(int(custom.dt*1000000));
  };
  return 0;
}
