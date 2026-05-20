// G1 Hand SDK Control Example
//
//   Motor_real = weight * Hand_SDK + (1 - weight) * G1_Cmd
//
// Demonstrates how a user process injects commands for the 4 hand motors
// into ai_sport via the DDS topic `rt/hand_sdk`. This example takes full
// control of the hand (weight = 1.0) and toggles tau between +0.3 and
// -0.3 every second to alternate between "close" and "open".
//
//
// Run:
//   ./hand_sdk_example              # use default network interface
//   ./hand_sdk_example eth0         # specify a network interface

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/go2/MotorCmds_.hpp>

#include <algorithm>
#include <chrono>
#include <thread>

class HandSdk {
public:
    static constexpr int kMotorNum = 4;

    explicit HandSdk(const std::string& topic = "rt/hand_sdk")
        : publisher_(std::make_shared<unitree::robot::ChannelPublisher<
              unitree_go::msg::dds_::MotorCmds_>>(topic)) {
        publisher_->InitChannel();
        msg_.cmds().resize(kMotorNum);
    }

    // weight is in [0, 1] and is encoded as `weight * 100` (uint8) in cmds[0].mode.
    float weight() const {
        return std::clamp(msg_.cmds()[0].mode() / 100.f, 0.f, 1.f);
    }
    void set_weight(float w) {
        msg_.cmds()[0].mode(
            static_cast<uint8_t>(std::clamp(w, 0.f, 1.f) * 100.f));
    }

    // Positive tau closes the hand, negative tau opens it.
    void set_tau(float tau) {
        for (int i = 0; i < kMotorNum; ++i) {
            msg_.cmds()[i].tau(tau);
        }
    }

    void write() { publisher_->Write(msg_); }

private:
    std::shared_ptr<unitree::robot::ChannelPublisher<
        unitree_go::msg::dds_::MotorCmds_>> publisher_;
    unitree_go::msg::dds_::MotorCmds_ msg_;
};

int main(int argc, char** argv) {
    if (argc > 1) {
        unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
    } else {
        unitree::robot::ChannelFactory::Instance()->Init(0);
    }

    HandSdk hand_sdk;
    hand_sdk.set_weight(1.0f);

    float tau = 0.3f;
    while (true) {
        tau = -tau;
        hand_sdk.set_tau(tau);
        hand_sdk.write();
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}
