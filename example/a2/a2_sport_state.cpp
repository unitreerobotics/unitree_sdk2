

#include <cmath>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>

#define TOPIC_HIGHSTATE "rt/sportmodestate"
using namespace unitree::common;

const std::vector<std::string> FSM_STATE_STR = {
    "PASSIVE",
    "STAND_DOWN",
    "STAND_UP",
    "DEFAULT_MODE",
    "RUNNING_MODE",
    "CLIMB_MODE",
    "LEFT_SIDE_GAIT",
    "RIGHT_SIDE_GAIT",
    "HANDSTAND",
    "BIPED_STAND",
    "FRONT_FLIP",
    "BACK_FLIP",
    "RECOVERY",
    "BASE_HEIGHT_CTRL"};

class Custom
{
public:
    Custom()
    {
        suber.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::SportModeState_>(TOPIC_HIGHSTATE));
        suber->InitChannel(std::bind(&Custom::HighStateHandler, this, std::placeholders::_1), 1);
    };

    void HighStateHandler(const void *message)
    {
        state = *(unitree_go::msg::dds_::SportModeState_ *)message;

        std::cout << "Position: " << state.position()[0] << ", " << state.position()[1] << ", " << state.position()[2] << std::endl;
        std::cout << "Velocity: " << state.velocity()[0] << ", " << state.velocity()[1] << ", " << state.velocity()[2] << std::endl;
        std::cout << "Mode: " << FSM_STATE_STR[int(state.mode())] << std::endl;
        std::cout << "Progress: " << state.progress() << std::endl;

        // std::cout << "IMU rpy: " << state.imu_state().rpy()[0] << ", " << state.imu_state().rpy()[1] << ", " << state.imu_state().rpy()[2] << std::endl;
    };

    unitree_go::msg::dds_::SportModeState_ state;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::SportModeState_> suber;
    float dt = 0.05; // 控制步长0.001~0.01
};

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
        exit(-1);
    }

    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    Custom custom;

    while (1)
    {
        sleep(10);
    }
    return 0;
}
