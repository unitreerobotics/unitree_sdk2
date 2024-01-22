#pragma once

#include <iostream>
#include <stdint.h>
#include <string>

#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include <unitree/common/thread/thread.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>

#include "base_state.h"
#include "data_buffer.hpp"
#include "motors.hpp"

static const std::string kTopicLowCommand = "rt/lowcmd";
static const std::string kTopicLowState = "rt/lowstate";

class HumanoidExample {
public:
  HumanoidExample(const std::string &networkInterface = "") {
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);
    std::cout << "Initialize channel factory." << std::endl;

    lowcmd_publisher_.reset(
        new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(
            kTopicLowCommand));
    lowcmd_publisher_->InitChannel();
    command_writer_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "command_writer", UT_CPU_ID_NONE, 2000,
        &HumanoidExample::LowCommandWriter, this);

    lowstate_subscriber_.reset(
        new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
            kTopicLowState));
    lowstate_subscriber_->InitChannel(
        std::bind(&HumanoidExample::LowStateHandler, this,
                  std::placeholders::_1),
        1);
    int control_period_us = control_dt_ * 1e6;
    control_thread_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, control_period_us, &HumanoidExample::Control,
        this);

    int report_period_us = report_dt_ * 1e6;
    report_rpy_ptr_ = unitree::common::CreateRecurrentThreadEx(
        "report_rpy", UT_CPU_ID_NONE, report_period_us,
        &HumanoidExample::ReportRPY, this);
  }

  ~HumanoidExample() = default;

  void LowCommandWriter() {
    unitree_go::msg::dds_::LowCmd_ low_cmd;
    PackOneLowCommand pack_one_low_command;
    const std::shared_ptr<const MotorCommand> mc_tmp_ptr =
        motor_command_buffer_.GetData();
    if (mc_tmp_ptr) {
      for (int i = 0; i < kNumMotors; ++i) {
        if (IsWeakMotor(i)) {
          pack_one_low_command.motor_command[i].mode = 0x01;
        } else {
          pack_one_low_command.motor_command[i].mode = 0x0A;
        }
        pack_one_low_command.motor_command[i].tau = mc_tmp_ptr->tau_ff.at(i);
        pack_one_low_command.motor_command[i].q = mc_tmp_ptr->q_ref.at(i);
        pack_one_low_command.motor_command[i].dq = mc_tmp_ptr->dq_ref.at(i);
        pack_one_low_command.motor_command[i].kp = mc_tmp_ptr->kp.at(i);
        pack_one_low_command.motor_command[i].kd = mc_tmp_ptr->kd.at(i);
      }
      pack_one_low_command.crc =
          Crc32Core((uint32_t *)&pack_one_low_command,
                    (sizeof(PackOneLowCommand) >> 2) - 1);
      LowCmd2DDS(pack_one_low_command, low_cmd);
      // easy_dds_.WriteMessage<unitree_go::msg::dds_::LowCmd_>(kTopicLowCommand,
      //                                                        low_cmd);
      lowcmd_publisher_->Write(low_cmd);
    }
  }

  void LowStateHandler(const void *message) {
    unitree_go::msg::dds_::LowState_ low_state =
        *(unitree_go::msg::dds_::LowState_ *)message;

    RecordMotorState(low_state);
    RecordBaseState(low_state);
  }

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms_tmp_ptr =
        motor_state_buffer_.GetData();

    if (ms_tmp_ptr) {
      time_ += control_dt_;
      time_ = std::clamp(time_, 0.f, init_duration_);
      float ratio = time_ / init_duration_;
      for (int i = 0; i < kNumMotors; ++i) {
        motor_command_tmp.kp.at(i) = IsWeakMotor(i) ? kp_low_ : kp_high_;
        motor_command_tmp.kd.at(i) = IsWeakMotor(i) ? kd_low_ : kd_high_;
        motor_command_tmp.dq_ref.at(i) = 0.f;
        motor_command_tmp.tau_ff.at(i) = 0.f;

        float q_des = 0.f;
        if (i == JointIndex::kLeftHipPitch || i == JointIndex::kRightHipPitch) {
          q_des = hip_pitch_init_pos_;
        }
        if (i == JointIndex::kLeftKnee || i == JointIndex::kRightKnee) {
          q_des = knee_init_pos_;
        }
        if (i == JointIndex::kLeftAnkle || i == JointIndex::kRightAnkle) {
          q_des = ankle_init_pos_;
        }
        if (i == JointIndex::kLeftShoulderPitch ||
            i == JointIndex::kRightShoulderPitch) {
          q_des = shoulder_pitch_init_pos_;
        }

        q_des = (q_des - ms_tmp_ptr->q.at(i)) * ratio + ms_tmp_ptr->q.at(i);
        motor_command_tmp.q_ref.at(i) = q_des;
      }

      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }

  void ReportRPY() {
    const std::shared_ptr<const BaseState> bs_tmp_ptr =
        base_state_buffer_.GetData();
    if (bs_tmp_ptr) {
      std::cout << "rpy: [" << bs_tmp_ptr->rpy.at(0) << ", "
                << bs_tmp_ptr->rpy.at(1) << ", " << bs_tmp_ptr->rpy.at(2) << "]"
                << std::endl;
    }
  }

private:
  void RecordMotorState(const unitree_go::msg::dds_::LowState_ &msg) {
    MotorState ms_tmp;
    for (int i = 0; i < kNumMotors; ++i) {
      ms_tmp.q.at(i) = msg.motor_state()[i].q();
      ms_tmp.dq.at(i) = msg.motor_state()[i].dq();
    }

    motor_state_buffer_.SetData(ms_tmp);
  }

  void RecordBaseState(const unitree_go::msg::dds_::LowState_ &msg) {
    BaseState bs_tmp;
    bs_tmp.omega = msg.imu_state().gyroscope();
    bs_tmp.rpy = msg.imu_state().rpy();

    base_state_buffer_.SetData(bs_tmp);
  }

  inline bool IsWeakMotor(int motor_index) {
    return motor_index == JointIndex::kLeftAnkle ||
           motor_index == JointIndex::kRightAnkle ||
           motor_index == JointIndex::kRightShoulderPitch ||
           motor_index == JointIndex::kRightShoulderRoll ||
           motor_index == JointIndex::kRightShoulderYaw ||
           motor_index == JointIndex::kRightElbow ||
           motor_index == JointIndex::kLeftShoulderPitch ||
           motor_index == JointIndex::kLeftShoulderRoll ||
           motor_index == JointIndex::kLeftShoulderYaw ||
           motor_index == JointIndex::kLeftElbow;
  }

  unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_>
      lowcmd_publisher_;
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>
      lowstate_subscriber_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<BaseState> base_state_buffer_;

  // control params
  float kp_low_ = 60.f;
  float kp_high_ = 200.f;
  float kd_low_ = 1.5f;
  float kd_high_ = 5.f;

  float control_dt_ = 0.01f;

  float hip_pitch_init_pos_ = -0.5f;
  float knee_init_pos_ = 1.f;
  float ankle_init_pos_ = -0.5f;
  float shoulder_pitch_init_pos_ = 0.4f;

  float time_ = 0.f;
  float init_duration_ = 10.f;

  float report_dt_ = 0.1f;

  // multithreading
  unitree::common::ThreadPtr command_writer_ptr_;
  unitree::common::ThreadPtr control_thread_ptr_;
  unitree::common::ThreadPtr report_rpy_ptr_;
};
