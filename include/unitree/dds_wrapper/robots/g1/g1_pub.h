// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <eigen3/Eigen/Dense>
#include <unitree/dds_wrapper/common/Publisher.h>
#include "unitree/dds_wrapper/common/crc.h"
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/MotorCmds_.hpp>


namespace unitree
{
namespace robot
{
namespace g1
{
namespace publisher
{

class LowState : public RealTimePublisher<unitree_hg::msg::dds_::LowState_>
{
public:
    LowState(std::string topic = "rt/lowstate") : RealTimePublisher<MsgType>(topic) 
    {
    }
    std::shared_ptr<unitree::common::UnitreeJoystick> joystick = nullptr;

private:
    void pre_communication() override {
      if (joystick) {
        auto data = joystick->combine();
        memcpy(&msg_.wireless_remote()[0], &data, sizeof(unitree::common::REMOTE_DATA_RX));
      }
      msg_.crc() = crc32_core((uint32_t*)&msg_, (sizeof(MsgType)>>2)-1);
    }
  };

class LowCmd : public RealTimePublisher<unitree_hg::msg::dds_::LowCmd_>
{
public:
    LowCmd(std::string topic = "rt/lowcmd") : RealTimePublisher<MsgType>(topic) 
    {
    }

private:
    /**
     * @brief Something before sending the message.
     */
    void pre_communication() override {
        msg_.crc() = crc32_core((uint32_t*)&msg_, (sizeof(MsgType)>>2)-1);
    }
};

class ArmSdk : public RealTimePublisher<unitree_hg::msg::dds_::LowCmd_>
{
public:
    ArmSdk(std::string topic = "rt/arm_sdk") : RealTimePublisher<MsgType>(topic) {}

    // Enable arm sdk
    void weight(float coe)
    {
        msg_.motor_cmd()[29].q() = std::clamp(coe, 0.0f, 1.0f);
    }

    float weight () const
    {
        return msg_.motor_cmd()[29].q();
    }
};


}; // namespace publisher
}; // namespace g1
}; // namespace robot
}; // namespace unitree