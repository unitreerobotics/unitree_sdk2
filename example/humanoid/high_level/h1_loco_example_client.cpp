#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#include "unitree/robot/channel/channel_subscriber.hpp"
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/robot/h1/loco/h1_loco_client.hpp>

constexpr float kPi_2 = 1.57079632;
unitree_go::msg::dds_::LowState_ state;

enum JointIndex {
  // Right arm
  kRightShoulderPitch = 12,
  kRightShoulderRoll = 13,
  kRightShoulderYaw = 14,
  kRightElbow = 15,
  // Left arm
  kLeftShoulderPitch = 16,
  kLeftShoulderRoll = 17,
  kLeftShoulderYaw = 18,
  kLeftElbow = 19,
};

const static std::vector<int> arm_joints = {
    JointIndex::kLeftShoulderPitch,  JointIndex::kLeftShoulderRoll,
    JointIndex::kLeftShoulderYaw,    JointIndex::kLeftElbow,
    JointIndex::kRightShoulderPitch, JointIndex::kRightShoulderRoll,
    JointIndex::kRightShoulderYaw,   JointIndex::kRightElbow};

void LowStateHandler(const void *message) {
  state = *(unitree_go::msg::dds_::LowState_ *)message;
}

std::vector<float> GetArmPosFromDds(unitree_go::msg::dds_::LowState_ state) {
  std::vector<float> vec = std::vector<float>(8, 0.f);
  for (int i = 0; i < 8; ++i) {
    vec.at(i) = state.motor_state().at(arm_joints.at(i)).q();
  }
  return vec;
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    exit(-1);
  }

  // set up client
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::h1::H1LocoClient client;

  client.Init();
  client.SetTimeout(10.0f);

  // set up feedback
  unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_>
      lowstate_subscriber;
  lowstate_subscriber.reset(
      new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::LowState_>(
          "rt/lowstate"));
  lowstate_subscriber->InitChannel(
      std::bind(&LowStateHandler, std::placeholders::_1), 1);

  // arm ctrl related variables
  std::vector<float> init_pos = std::vector<float>(8, 0.f);
  std::vector<float> jpos = std::vector<float>(8, 0.f);
  std::vector<float> target_pos{0.f, kPi_2,  0.f, kPi_2,
                                0.f, -kPi_2, 0.f, kPi_2};
  std::vector<float> dq = std::vector<float>(8, 0.f);
  std::vector<float> kp = std::vector<float>(8, 80.f);
  std::vector<float> kd = std::vector<float>(8, 2.f);
  std::vector<float> tau_ff = std::vector<float>(8, 0.f);

  float weight = 0.f;
  float weight_rate = 0.6f;

  float control_dt = 0.02f;
  float max_joint_velocity = 0.5f;

  float delta_weight = weight_rate * control_dt;
  float max_joint_delta = max_joint_velocity * control_dt;
  auto sleep_time =
      std::chrono::milliseconds(static_cast<int>(control_dt / 0.001f));

  float init_stop_time = 2.f;
  int init_stop_num_steps = static_cast<int>(init_stop_time / control_dt);
  float arm_move_time = 5.f;
  int arm_move_num_steps = static_cast<int>(arm_move_time / control_dt);

  // start main loop
  while (true) {
    if (std::cin.peek() != EOF) {
      char key;
      std::cin >> key;
      if (key == 'j') {
        client.EnableCtrl();
      }
      if (key == 'k') {
        client.Stop();
      }
      if (key == 'o') {
        client.DecreaseSwingHeight();
      }
      if (key == 'p') {
        client.IncreaseSwingHeight();
      }
      if (key == 'l') {
        client.StandSwitch();
      }
      if (key == 'w') {
        client.Move(0.5, 0.0, 0.0);
      }
      if (key == 'a') {
        client.Move(0., 0.5, 0.0);
      }
      if (key == 's') {
        client.Move(-0.5, 0.0, 0.0);
      }
      if (key == 'd') {
        client.Move(0., -0.5, 0.0);
      }
      if (key == 'q') {
        client.Move(0., 0.0, 0.5);
      }
      if (key == 'e') {
        client.Move(0., 0.0, -0.5);
      }
      if (key == 'u') {
        // enable arm ctrl
        weight = 0.f;
        for (int i = 0; i < init_stop_num_steps; ++i) {
          weight += delta_weight;
          weight = std::clamp(weight, 0.f, 1.f);

          client.ArmCtrl(init_pos, dq, kp, kd, tau_ff, weight * weight);

          std::this_thread::sleep_for(sleep_time);
        }
      }
      if (key == 'i') {
        // diable arm ctrl
        weight = 1.f;
        jpos = GetArmPosFromDds(state);
        for (int i = 0; i < init_stop_num_steps; ++i) {
          weight -= delta_weight;
          weight = std::clamp(weight, 0.f, 1.f);

          client.ArmCtrl(jpos, dq, kp, kd, tau_ff, weight);

          std::this_thread::sleep_for(sleep_time);
        }
      }
      if (key == 'm') {
        // move arm to pos 0
        jpos = GetArmPosFromDds(state);
        for (int i = 0; i < arm_move_num_steps; ++i) {
          for (int j = 0; j < init_pos.size(); ++j) {
            jpos.at(j) += std::clamp(init_pos.at(j) - jpos.at(j),
                                     -max_joint_delta, max_joint_delta);
          }

          client.ArmCtrl(jpos, dq, kp, kd, tau_ff, 1.f);

          std::this_thread::sleep_for(sleep_time);
        }
      }
      if (key == 'n') {
        // move arm to pos 1
        jpos = GetArmPosFromDds(state);
        for (int i = 0; i < arm_move_num_steps; ++i) {
          for (int j = 0; j < init_pos.size(); ++j) {
            jpos.at(j) += std::clamp(target_pos.at(j) - jpos.at(j),
                                     -max_joint_delta, max_joint_delta);
          }

          client.ArmCtrl(jpos, dq, kp, kd, tau_ff, 1.f);

          std::this_thread::sleep_for(sleep_time);
        }
      }
    }
  }

  return 0;
}