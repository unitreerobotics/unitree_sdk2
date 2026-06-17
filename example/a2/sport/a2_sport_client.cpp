#include <iostream>
#include <string>
#include <chrono>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <stdexcept>
#include <cmath>
#include <unitree/robot/a2/sport/sport_client.hpp>

using namespace std;

struct TestOption
{
    std::string name;
    int id;
};

const vector<TestOption> option_list =
    {
        {"damp", 0},
        {"balance_stand", 1},
        {"stop_move", 2},
        {"stand_down", 3},
        {"recovery_stand", 4},
        {"move", 5},
        {"switch_gait", 6},
        {"speed_level", 7},
        {"get_state", 8},
        {"recovery_switch", 9},
        {"body_height", 10},
        {"stand_up", 11},
        
        // Caution:  test in open area
        {"enter_leftside_gait", 12},
        {"exit_leftside_gait", 13},
        {"enter_handstand", 14},
        {"exit_handstand", 15},
        {"front_flip", 16},
        {"back_flip", 17},
        {"pose", 18},
        {"euler", 19},
        {"reset_estimator", 20},
        {"square_trajectory", 21},
        {"circle_trajectory", 22},
        
};

int ConvertToInt(const std::string &str)
{
    try
    {
        std::stoi(str);
        return std::stoi(str);
    }
    catch (const std::invalid_argument &)
    {
        return -1;
    }
    catch (const std::out_of_range &)
    {
        return -1;
    }
}

class UserInterface
{
public:
    UserInterface() {};
    ~UserInterface() {};

    void terminalHandle()
    {
        std::string input;
        std::getline(std::cin, input);

        if (input.compare("list") == 0)
        {
            for (TestOption option : option_list)
            {
                std::cout << option.name << ", id: " << option.id << std::endl;
            }
        }

        for (TestOption option : option_list)
        {
            if (input.compare(option.name) == 0 || ConvertToInt(input) == option.id)
            {
                test_option_->id = option.id;
                test_option_->name = option.name;
                std::cout << "Test: " << test_option_->name << ", test_id: " << test_option_->id << std::endl;
            }
        }
    };

    TestOption *test_option_;
};

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

    TestOption test_option;
    test_option.id = 1;

    unitree::robot::a2::SportClient sport_client;
    sport_client.SetTimeout(25.0f);
    sport_client.Init();

    UserInterface user_interface;
    user_interface.test_option_ = &test_option;

    std::cout << "Input \"list \" to list all test option ..." << std::endl;
    long res_count = 0;
    while (1)
    {
        auto time_start_trick = std::chrono::high_resolution_clock::now();
        static const constexpr auto dt = std::chrono::microseconds(20000); // 50Hz

        user_interface.terminalHandle();

        int res = 1;
        if (test_option.id == 0)
        {
            res = sport_client.Damp();
        }
        else if (test_option.id == 1)
        {
            res = sport_client.BalanceStand();
        }
        else if (test_option.id == 2)
        {
            res = sport_client.StopMove();
        }
        else if (test_option.id == 3)
        {
            res = sport_client.StandDown();
        }
        else if (test_option.id == 4)
        {
            res = sport_client.RecoveryStand();
        }
        else if (test_option.id == 5)
        {
            res = sport_client.Move(0.0, 0.0, 0.5);
        }
        else if (test_option.id == 6)
        {
            res = sport_client.SwitchGait(0);
        }
        else if (test_option.id == 7)
        {
            res = sport_client.SpeedLevel(1);
        }
        else if (test_option.id == 8)
        {
            std::map<std::string, std::string> state_map;
            res = sport_client.GetState(state_map);
            std::cout << "fsm_id: " <<  state_map["fsm_id"] << std::endl;
            std::cout << "fsm_name: " <<  state_map["fsm_name"] << std::endl;
            std::cout << "speed_level: " << state_map["speed_level"] << std::endl;
            std::cout << "auto_recovery_switch: " << state_map["auto_recovery_switch"] << std::endl;
            std::cout << "process_state: " << state_map["process_state"] << std::endl;
        }
        else if (test_option.id == 9)
        {
            res = sport_client.SetAutoRecovery(0);
        }
        else if (test_option.id == 10)
        {
            res = sport_client.BodyHeight(0.3f);
        }
        else if (test_option.id == 11)
        {
            res = sport_client.StandUp();
        }
        else if (test_option.id == 12)
        {
            res = sport_client.LeftSideGait(1);
        }
        else if (test_option.id == 13)
        {
            res = sport_client.LeftSideGait(0);
        }
        else if (test_option.id == 14)
        {
            res = sport_client.HandStand(1);
        }
        else if (test_option.id == 15)
        {
            res = sport_client.HandStand(0);
        }
        else if (test_option.id == 16)
        {
            res = sport_client.FrontFlip();
        }
        else if (test_option.id == 17)
        {
            res = sport_client.BackFlip();
        }
        else if (test_option.id == 18)
        {
            res = sport_client.BodyPosition(0.2f, 0.2f, -0.2f, 0.2f);
        }
        else if (test_option.id == 19)
        {
            res = sport_client.Euler(0.2f, 0.3f, 0.3f);
        }
        else if (test_option.id == 20)
        {
            res = sport_client.ResetEstimator();
        }
        else if (test_option.id == 21)
        {
            // 矩形轨迹: 1.0m x 1.0m, 带加减速过程的密集路径点
            std::vector<unitree::robot::a2::PathPoint> path;
            unitree::robot::a2::PathPoint pt;

            // ===== 边1: (0,0)→(1,0), 向前 1m, 0-4s =====
            pt.t_from_start = 0.0f;  pt.x = 0.00f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 0.5f;  pt.x = 0.03f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 1.0f;  pt.x = 0.12f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 1.5f;  pt.x = 0.28f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 2.0f;  pt.x = 0.45f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 2.5f;  pt.x = 0.62f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 3.0f;  pt.x = 0.78f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 3.5f;  pt.x = 0.94f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 4.0f;  pt.x = 1.00f; pt.y = 0.0f; pt.yaw = 0.0f; path.push_back(pt);

            // ===== 边2: (1,0)→(1,1), 左移 1m, 4-8s =====
            pt.t_from_start = 4.5f;  pt.x = 1.00f; pt.y = 0.03f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 5.0f;  pt.x = 1.00f; pt.y = 0.12f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 5.5f;  pt.x = 1.00f; pt.y = 0.28f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 6.0f;  pt.x = 1.00f; pt.y = 0.45f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 6.5f;  pt.x = 1.00f; pt.y = 0.62f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 7.0f;  pt.x = 1.00f; pt.y = 0.78f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 7.5f;  pt.x = 1.00f; pt.y = 0.94f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 8.0f;  pt.x = 1.00f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);

            // ===== 边3: (1,1)→(0,1), 后退 1m, 8-13s =====
            pt.t_from_start = 8.5f;  pt.x = 0.97f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 9.0f;  pt.x = 0.90f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 9.5f;  pt.x = 0.78f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 10.0f; pt.x = 0.62f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 10.5f; pt.x = 0.45f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 11.0f; pt.x = 0.28f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 11.5f; pt.x = 0.12f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 12.0f; pt.x = 0.03f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 13.0f; pt.x = 0.00f; pt.y = 1.00f; pt.yaw = 0.0f; path.push_back(pt);

            // ===== 边4: (0,1)→(0,0), 右移 1m, 13-16s =====
            pt.t_from_start = 13.5f; pt.x = 0.00f; pt.y = 0.94f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 14.0f; pt.x = 0.00f; pt.y = 0.78f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 14.5f; pt.x = 0.00f; pt.y = 0.55f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 15.0f; pt.x = 0.00f; pt.y = 0.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 15.5f; pt.x = 0.00f; pt.y = 0.00f; pt.yaw = 0.0f; path.push_back(pt);
            pt.t_from_start = 16.0f; pt.x = 0.00f; pt.y = 0.00f; pt.yaw = 0.0f; path.push_back(pt);

            res = sport_client.Trajectory(path, 1);
        }
        else if (test_option.id == 22)
        {
            // 圆形轨迹: 半径1.0m, 圆心(0.0,1.0), 逆时针
            const float R = 1.0f, cx = 0.0f, cy = 1.0f;
            const float TOTAL_TIME = 20.0f;          // <-- 修改此处调整总时长(s)
            const float T_ACCEL  = 0.25f * TOTAL_TIME;  // 加速段
            const float T_CRUISE = 0.50f * TOTAL_TIME;  // 匀速段
            const float T_DECEL  = 0.25f * TOTAL_TIME;  // 减速段
            const int   N = 36;
            std::vector<unitree::robot::a2::PathPoint> path;

            for (int i = 0; i <= N; ++i)
            {
                float frac = (float)i / N;
                float t;
                if (frac <= 0.25f)
                    t = T_ACCEL * (frac / 0.25f);
                else if (frac <= 0.75f)
                    t = T_ACCEL + T_CRUISE * ((frac - 0.25f) / 0.5f);
                else
                    t = T_ACCEL + T_CRUISE + T_DECEL * ((frac - 0.75f) / 0.25f);

                float angle = -M_PI_2 + 2.0f * M_PI * frac;

                unitree::robot::a2::PathPoint pt;
                pt.t_from_start = t;
                pt.x = cx + R * cosf(angle);
                pt.y = cy + R * sinf(angle);
                pt.yaw = 0.0f;
                path.push_back(pt);
            }

            res = sport_client.Trajectory(path, 1);
        }


        if (res < 0)
        {
            res_count += 1;
            std::cout << "Request error for: " << option_list[test_option.id].name << ", code: " << res << ", count: " << res_count << std::endl;
        }
        else
        {
            res_count = 0;
            std::cout << "Request successed: " << option_list[test_option.id].name << ", code: " << res << std::endl;
        }
        std::this_thread::sleep_until(time_start_trick + dt);
    }
    return 0;
}