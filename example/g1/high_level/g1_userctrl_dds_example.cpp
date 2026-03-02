#include <array>
#include <chrono>
#include <iostream>
#include <thread>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include "unitree/robot/g1/loco/g1_loco_api.hpp"
#include "unitree/robot/g1/loco/g1_loco_client.hpp"

static const std::string kTopicUserCtrl = "rt/user_lowcmd";
static const std::string kTopicState = "rt/lowstate";

enum JointIndex {
    // Left leg
    kLeftHipPitch,
    kLeftHipRoll,
    kLeftHipYaw,
    kLeftKnee,
    kLeftAnkle,
    kLeftAnkleRoll,

    // Right leg
    kRightHipPitch,
    kRightHipRoll,
    kRightHipYaw,
    kRightKnee,
    kRightAnkle,
    kRightAnkleRoll,

    kWaistYaw,
    kWaistRoll,
    kWaistPitch,

    // Left arm
    kLeftShoulderPitch,
    kLeftShoulderRoll,
    kLeftShoulderYaw,
    kLeftElbow,
    kLeftWristRoll,
    kLeftWristPitch,
    kLeftWristYaw,
    // Right arm
    kRightShoulderPitch,
    kRightShoulderRoll,
    kRightShoulderYaw,
    kRightElbow,
    kRightWristRoll,
    kRightWristPitch,
    kRightWristYaw,

    kNotUsedJoint,
    kNotUsedJoint1,
    kNotUsedJoint2,
    kNotUsedJoint3,
    kNotUsedJoint4,
    kNotUsedJoint5
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    exit(-1);
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_>
      user_ctrl_publisher;
  unitree_hg::msg::dds_::LowCmd_ msg;

  user_ctrl_publisher.reset(
      new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(
          kTopicUserCtrl));
  user_ctrl_publisher->InitChannel();

  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_>
      low_state_subscriber;

  // create subscriber
  unitree_hg::msg::dds_::LowState_ state_msg;
  low_state_subscriber.reset(
      new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
          kTopicState));
  low_state_subscriber->InitChannel([&](const void *msg) {
        auto s = ( const unitree_hg::msg::dds_::LowState_* )msg;
        memcpy( &state_msg, s, sizeof( unitree_hg::msg::dds_::LowState_ ) );
  }, 1);

  unitree::robot::g1::LocoClient client;
  client.Init();
  client.SetTimeout(5.f);
  int current_fsm_id;
  client.GetFsmId(current_fsm_id);
  if(current_fsm_id != 1){
    std::cout << "Current fsm is not PASSIVE, exitting ..." <<std::endl;
    exit(-1);
  }

  float kp = 60.f;
  float kd = 1.5f;
  float dq = 0.f;
  float tau_ff = 0.f;

  float control_dt = 0.02f;
  float max_joint_velocity = 0.5f;

  float max_joint_delta = max_joint_velocity * control_dt;
  auto sleep_time =
      std::chrono::milliseconds(static_cast<int>(control_dt / 0.001f));

  std::array<float, 29> init_pos{-0.1, 0, 0, 0.3, -0.2, 0,
                                 -0.1, 0, 0, 0.3, -0.2, 0,
                                 0, 0, 0,
                                 0.2, 0.2, 0, 0.9, 0, 0, 0,
                                 0.2, -0.2, 0, 0.9, 0, 0, 0};

  std::array<float, 29> target_pos = {-1.0, 0, 0, 1.5, 0.2, 0,
                                      -1.0, 0, 0, 1.5, 0.2, 0,
                                      0, 0, 0,
                                      0, 0.2, 0, 0, 0, 0, 0,
                                      0, -0.2, 0, 0, 0, 0, 0};

  // wait for init
  std::cout << "Press ENTER to init user control ...";
  std::cin.get();
  for (int j = 0; j < init_pos.size(); ++j) {
      msg.motor_cmd().at(j).q(0);
      msg.motor_cmd().at(j).dq(0);
      msg.motor_cmd().at(j).kp(0);
      msg.motor_cmd().at(j).kd(kd);
      msg.motor_cmd().at(j).tau(0);
    }
  user_ctrl_publisher->Write(msg);

  client.SwitchToUserCtrl();

  // get current joint position
  std::array<float, 29> current_jpos{};
  std::cout<<"Current joint position: ";
  for (int i = 0; i < init_pos.size(); ++i) {
	current_jpos.at(i) = state_msg.motor_state().at(i).q();
	std::cout << current_jpos.at(i) << " ";
  }
  std::cout << std::endl;

  // set init pos
  std::cout << "Initailizing motors ...";
  float init_time = 2.0f;
  int init_time_steps = static_cast<int>(init_time / control_dt);

  for (int i = 0; i < init_time_steps; ++i) {
    
	  float phase = 1.0 * i / init_time_steps;
	  std::cout << "Phase: " << phase << std::endl;

    // set control joints
    for (int j = 0; j < init_pos.size(); ++j) {
      msg.motor_cmd().at(j).q(init_pos.at(j) * phase + current_jpos.at(j) * (1 - phase));
      msg.motor_cmd().at(j).dq(dq);
      msg.motor_cmd().at(j).kp(kp);
      msg.motor_cmd().at(j).kd(kd);
      msg.motor_cmd().at(j).tau(tau_ff);
    }

    // send dds msg
    user_ctrl_publisher->Write(msg);

    // sleep
    std::this_thread::sleep_for(sleep_time);
  }

  std::cout << "Done!" << std::endl;

  // wait for control
  std::cout << "Press ENTER to start user control ..." << std::endl;
  std::cin.get();

  // start control
  std::cout << "Start user control!" << std::endl;
  float period = 5.f;
  int num_time_steps = static_cast<int>(period / control_dt);

  std::array<float, 29> current_jpos_des{};
  for (int i = 0; i < init_pos.size(); ++i) {
	  current_jpos_des.at(i) = state_msg.motor_state().at(i).q();
  }

  // lift arms up
  for (int i = 0; i < num_time_steps; ++i) {
    // update jpos des
    for (int j = 0; j < init_pos.size(); ++j) {
      current_jpos_des.at(j) +=
          std::clamp(target_pos.at(j) - current_jpos_des.at(j),
                     -max_joint_delta, max_joint_delta);
    }

    // set control joints
    for (int j = 0; j < init_pos.size(); ++j) {
      msg.motor_cmd().at(j).q(current_jpos_des.at(j));
      msg.motor_cmd().at(j).dq(dq);
      msg.motor_cmd().at(j).kp(kp);
      msg.motor_cmd().at(j).kd(kd);
      msg.motor_cmd().at(j).tau(tau_ff);
    }

    // send dds msg
    user_ctrl_publisher->Write(msg);

    // sleep
    std::this_thread::sleep_for(sleep_time);
  }

  // put arms down
  for (int i = 0; i < num_time_steps; ++i) {
    // update jpos des
    for (int j = 0; j < init_pos.size(); ++j) {
      current_jpos_des.at(j) +=
          std::clamp(init_pos.at(j) - current_jpos_des.at(j), -max_joint_delta,
                     max_joint_delta);
    }

    // set control joints
    for (int j = 0; j < init_pos.size(); ++j) {
      msg.motor_cmd().at(j).q(current_jpos_des.at(j));
      msg.motor_cmd().at(j).dq(dq);
      msg.motor_cmd().at(j).kp(kp);
      msg.motor_cmd().at(j).kd(kd);
      msg.motor_cmd().at(j).tau(tau_ff);
    }

    // send dds msg
    user_ctrl_publisher->Write(msg);

    // sleep
    std::this_thread::sleep_for(sleep_time);
  }

  // stop control
  std::cout << "Stoping user control ...";
  client.SwitchToInternalCtrl(unitree::robot::g1::InternalFsmMode::LAST);
  
  std::cout << "Done!" << std::endl;

  return 0;
}
