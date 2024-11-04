#include <array>
#include <chrono>
#include <iostream>
#include <thread>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

static const std::string kTopicArmSDK = "rt/arm_sdk";
constexpr float kPi = 3.141592654;
constexpr float kPi_2 = 1.57079632;

enum JointIndex {
  // Left leg
  kLeftHipYaw = 0,
  kLeftHipPitch = 1,
  kLeftHipRoll = 2,
  kLeftKnee = 3,
  kLeftAnkle = 4,
  kLeftAnkleRoll = 5,
  // Right leg
  kRightHipYaw = 6,
  kRightHipPitch = 7,
  kRightHipRoll = 8,
  kRightKnee = 9,
  kRightAnkle = 10,
  kRightAnkleRoll = 11,

  kWaistYaw = 12,

  // Left arm
  kLeftShoulderPitch = 13,
  kLeftShoulderRoll = 14,
  kLeftShoulderYaw = 15,
  kLeftElbow = 16,
  kLeftWistRoll = 17,
  kLeftWistPitch = 18,
  kLeftWistYaw = 19,
  // Right arm
  kRightShoulderPitch = 20,
  kRightShoulderRoll = 21,
  kRightShoulderYaw = 22,
  kRightElbow = 23,
  kRightWistRoll = 24,
  kRightWistPitch = 25,
  kRightWistYaw = 26,

  kNotUsedJoint = 27,
  kNotUsedJoint1 = 28,
  kNotUsedJoint2 = 29,
  kNotUsedJoint3 = 30,
  kNotUsedJoint4 = 31,
  kNotUsedJoint5 = 32,
  kNotUsedJoint6 = 33,
  kNotUsedJoint7 = 34
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    exit(-1);
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);

  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_>
      arm_sdk_publisher;
  unitree_hg::msg::dds_::LowCmd_ msg;

  arm_sdk_publisher.reset(
      new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(
          kTopicArmSDK));
  arm_sdk_publisher->InitChannel();

  std::array<JointIndex, 15> arm_joints = {
      JointIndex::kLeftShoulderPitch,  JointIndex::kLeftShoulderRoll,
      JointIndex::kLeftShoulderYaw,    JointIndex::kLeftElbow,
      JointIndex::kLeftWistRoll,       JointIndex::kLeftWistPitch,     JointIndex::kLeftWistYaw,       
      JointIndex::kRightShoulderPitch, JointIndex::kRightShoulderRoll,
      JointIndex::kRightShoulderYaw,   JointIndex::kRightElbow, 
      JointIndex::kRightWistRoll,      JointIndex::kRightWistPitch,    JointIndex::kRightWistYaw,       
      JointIndex::kWaistYaw};

  float weight = 0.f;
  float weight_rate = 0.2f;

  // TODO: 用数组来精细的设置每个关节的kp和kd
  std::array<float, 15> kp_array = { 120, 120, 80, 50, 50, 50, 50, 
                                    120, 120, 80, 50, 50, 50, 50, 
                                    200 };
  std::array<float, 15> kd_array = { 2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 
                                    2.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 
                                    2.0 };
  float dq = 0.f;
  float tau_ff = 0.f;

  float control_dt = 0.02f;
  float max_joint_velocity = 0.5f;

  float delta_weight = weight_rate * control_dt;
  float max_joint_delta = max_joint_velocity * control_dt;
  auto sleep_time =
      std::chrono::milliseconds(static_cast<int>(control_dt / 0.001f));

  // 修改初始位置，避免碰撞
  std::array<float, 15> init_pos{0.f, 0.3,  0.f, 0, 0, 0, 0,
                                     0.f, -0.3, 0.f, 0, 0, 0, 0,
                                     0.f};

  std::array<float, 15> target_pos = {0.f, kPi_2,  0.f, kPi_2, 0, 0, 0,
                                     0.f, -kPi_2, 0.f, kPi_2, 0, 0, 0,
                                     0.f};

  // wait for init
  std::cout << "Press ENTER to init arms ...";
  std::cin.get();

  // set init pos
  std::cout << "Initailizing arms ...";
  float init_time = 5.0f;
  int init_time_steps = static_cast<int>(init_time / control_dt);

  for (int i = 0; i < init_time_steps; ++i) {
    // increase weight
    weight += delta_weight;
    weight = std::clamp(weight, 0.f, 1.f);
    std::cout << weight << std::endl;

    // set weight
    msg.motor_cmd().at(JointIndex::kNotUsedJoint).q(weight * weight);

    // set control joints
    for (int j = 0; j < init_pos.size(); ++j) {
      msg.motor_cmd().at(arm_joints.at(j)).q(init_pos.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).dq(dq);
      msg.motor_cmd().at(arm_joints.at(j)).kp(kp_array.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).kd(kd_array.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).tau(tau_ff);
    }

    // send dds msg
    arm_sdk_publisher->Write(msg);

    // sleep
    std::this_thread::sleep_for(sleep_time);
  }

  std::cout << "Done!" << std::endl;

  // wait for control
  std::cout << "Press ENTER to start arm ctrl ..." << std::endl;
  std::cin.get();

  // start control
  std::cout << "Start arm ctrl!" << std::endl;
  float period = 5.f;
  int num_time_steps = static_cast<int>(period / control_dt);

  std::array<float, 15> current_jpos_des{0.f, 0.3,  0.f, 0, 0, 0, 0,
                                        0.f, -0.3, 0.f, 0, 0, 0, 0,
                                        0.f};

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
      msg.motor_cmd().at(arm_joints.at(j)).q(current_jpos_des.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).dq(dq);
      msg.motor_cmd().at(arm_joints.at(j)).kp(kp_array.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).kd(kd_array.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).tau(tau_ff);
    }

    // send dds msg
    arm_sdk_publisher->Write(msg);

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
      msg.motor_cmd().at(arm_joints.at(j)).q(current_jpos_des.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).dq(dq);
      msg.motor_cmd().at(arm_joints.at(j)).kp(kp_array.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).kd(kd_array.at(j));
      msg.motor_cmd().at(arm_joints.at(j)).tau(tau_ff);
    }

    // send dds msg
    arm_sdk_publisher->Write(msg);

    // sleep
    std::this_thread::sleep_for(sleep_time);
  }

  // stop control
  std::cout << "Stoping arm ctrl ...";
  float stop_time = 2.0f;
  int stop_time_steps = static_cast<int>(stop_time / control_dt);

  for (int i = 0; i < stop_time_steps; ++i) {
    // increase weight
    weight -= delta_weight;
    weight = std::clamp(weight, 0.f, 1.f);

    // set weight
    msg.motor_cmd().at(JointIndex::kNotUsedJoint).q(weight);

    // send dds msg
    arm_sdk_publisher->Write(msg);

    // sleep
    std::this_thread::sleep_for(sleep_time);
  }

  std::cout << "Done!" << std::endl;

  return 0;
}
