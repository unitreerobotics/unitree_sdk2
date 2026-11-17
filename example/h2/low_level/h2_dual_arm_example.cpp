#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "../../common/behavior_dir.hpp"

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace {

using LowCmd = unitree_hg::msg::dds_::LowCmd_;
using LowState = unitree_hg::msg::dds_::LowState_;
constexpr char kCmdTopic[] = "rt/lowcmd";
constexpr char kStateTopic[] = "rt/lowstate";
constexpr int kNumMotors = 31;
constexpr int kLeftShoulderPitch = 15;
constexpr int kRightWristYaw = 28;
constexpr int kHeadPitch = 29;
constexpr int kHeadYaw = 30;
constexpr double kControlDt = 0.002;
constexpr double kZeroDuration = 3.0;
// Replay the recorded trajectory at 60% of its original speed.
constexpr double kTrajectorySpeedScale = 0.6;

std::atomic<bool> g_stop{false};
void HandleSignal(int) { g_stop.store(true); }

const std::array<float, kNumMotors> kKp = {
    150, 150, 150, 250, 60, 90, 150, 150, 150, 250, 60, 90,
    200, 200, 200, 90, 60, 20, 60, 4, 4, 4, 90, 60, 20, 60,
    4, 4, 4, 30, 30};
const std::array<float, kNumMotors> kKd = {
    2, 2, 2, 2, .3f, .1f, 2, 2, 2, 2, .3f, .1f, 2.5f, 5, 5,
    2, 1, .4f, 1, .2f, .2f, .2f, 2, 1, .4f, 1, .2f, .2f, .2f, 1, 1};

inline uint32_t Crc32Core(uint32_t* ptr, uint32_t len) {
  uint32_t crc = 0xFFFFFFFF, polynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    uint32_t data = ptr[i], bit = 1u << 31;
    for (int j = 0; j < 32; ++j) {
      crc = (crc & 0x80000000) ? (crc << 1) ^ polynomial : crc << 1;
      if (data & bit) crc ^= polynomial;
      bit >>= 1;
    }
  }
  return crc;
}

class StateBuffer {
 public:
  void Set(const LowState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    state_ = state;
    ready_ = true;
  }
  bool Get(LowState& state) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!ready_) return false;
    state = state_;
    return true;
  }

 private:
  mutable std::mutex mutex_;
  LowState state_{};
  bool ready_ = false;
};

std::vector<std::array<float, 14>> LoadTrajectory(const std::string& path) {
  YAML::Node root = YAML::LoadFile(path);
  if (root["frame_rate"].as<int>() != 500) {
    throw std::runtime_error("motion.seq must use 500 Hz");
  }
  for (const auto& component : root["components"]) {
    if (component["content"].as<std::string>("") != "JointDisplacement") continue;
    std::vector<std::array<float, 14>> frames;
    for (const auto& frame : component["frames"]) {
      if (frame.size() != 14) throw std::runtime_error("trajectory frame is not 14-DOF");
      std::array<float, 14> values{};
      for (size_t i = 0; i < values.size(); ++i) values[i] = frame[i].as<float>();
      frames.push_back(values);
    }
    return frames;
  }
  throw std::runtime_error("JointDisplacement component not found");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "Usage: h2_dual_arm_example network_interface" << std::endl;
    return 0;
  }
  ::signal(SIGINT, HandleSignal);
  ::signal(SIGTERM, HandleSignal);

  try {
    const std::string network = argv[1];
    const std::string trajectory_path =
        unitree_example::ResolveBehaviorDir("h2") + "motion.seq";
    const auto trajectory = LoadTrajectory(trajectory_path);
    std::cout << "Loaded " << trajectory.size() << " frames from " << trajectory_path << std::endl;

    unitree::robot::ChannelFactory::Instance()->Init(0, network);
    auto motion_switcher = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
    motion_switcher->SetTimeout(5.0f);
    motion_switcher->Init();
    std::string form, mode;
    while (!g_stop.load() && motion_switcher->CheckMode(form, mode) == 0 && !mode.empty()) {
      std::cout << "Releasing motion service: " << mode << std::endl;
      if (motion_switcher->ReleaseMode() != 0) throw std::runtime_error("ReleaseMode failed");
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    auto publisher = std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(kCmdTopic);
    publisher->InitChannel();
    auto state_buffer = std::make_shared<StateBuffer>();
    auto subscriber = std::make_shared<unitree::robot::ChannelSubscriber<LowState>>(kStateTopic);
    subscriber->InitChannel([state_buffer](const void* message) {
      const auto& state = *static_cast<const LowState*>(message);
      if (state.crc() == Crc32Core((uint32_t*)&state, (sizeof(LowState) >> 2) - 1)) state_buffer->Set(state);
    }, 1);

    LowState state;
    while (!g_stop.load() && !state_buffer->Get(state)) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (g_stop.load()) return 0;

    LowCmd command{};
    double time = 0.0;
    while (!g_stop.load()) {
      if (!state_buffer->Get(state)) continue;
      command.mode_pr() = 0;
      command.mode_machine() = state.mode_machine();
      const double ratio = std::clamp(time / kZeroDuration, 0.0, 1.0);
      const size_t frame = time < kZeroDuration
          ? 0
          : std::min(static_cast<size_t>((time - kZeroDuration) *
                                         kTrajectorySpeedScale / kControlDt),
                     trajectory.size() - 1);
      for (int i = 0; i < kNumMotors; ++i) {
        auto& motor = command.motor_cmd().at(i);
        motor.mode(1);
        motor.tau(0.0f);
        motor.dq(0.0f);
        motor.kp(kKp[i]);
        motor.kd(kKd[i]);
        if (time < kZeroDuration) {
          motor.q(static_cast<float>((1.0 - ratio) * state.motor_state().at(i).q()));
        } else if (i >= kLeftShoulderPitch && i <= kRightWristYaw) {
          motor.q(trajectory[frame][static_cast<size_t>(i - kLeftShoulderPitch)]);
        } else {
          motor.q(0.0f);
        }
      }
      command.crc() = Crc32Core((uint32_t*)&command, (sizeof(LowCmd) >> 2) - 1);
      publisher->Write(command);
      time += kControlDt;
      if (time >= kZeroDuration +
                      trajectory.size() * kControlDt / kTrajectorySpeedScale) {
        time = 0.0;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    std::cout << "Stopped by Ctrl+C; low-level command loop exited." << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "h2_dual_arm_example failed: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
