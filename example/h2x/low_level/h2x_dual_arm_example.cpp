#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <shared_mutex>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>



static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;

const int H2X_NUM_MOTOR = 31;

// The compile-time macro points at the behaviors shipped with the source tree,
// the environment variable of the same name overrides it when it is set.
static std::string BehaviorDir() {
  std::string dir = H2X_DUAL_ARM_BEHAVIOR_DIR;
  const char *env = std::getenv("H2X_DUAL_ARM_BEHAVIOR_DIR");
  if (env != nullptr && env[0] != '\0') dir = env;
  if (!dir.empty() && dir.back() != '/') dir += '/';
  return dir;
}

template <typename T>
class DataBuffer {
 public:
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

 private:
  std::shared_ptr<T> data;
  std::shared_mutex mutex;
};

struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
};

struct MotorCommand {
  std::array<float, H2X_NUM_MOTOR> q_target = {};
  std::array<float, H2X_NUM_MOTOR> dq_target = {};
  std::array<float, H2X_NUM_MOTOR> kp = {};
  std::array<float, H2X_NUM_MOTOR> kd = {};
  std::array<float, H2X_NUM_MOTOR> tau_ff = {};
};

struct MotorState {
  std::array<float, H2X_NUM_MOTOR> q = {};
  std::array<float, H2X_NUM_MOTOR> dq = {};
};

enum PRorAB { PR = 0, AB = 1 };

enum H2XJointValidIndex {
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,
  LeftWristYaw = 21,
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,
  RightWristYaw = 28,
  HeadPitch = 29,
  HeadYaw = 30
};

// Stiffness for all H2X Joints: kp = tau_max / dq_sat, so a joint reaches its
// torque limit at dq_sat (1 ~ 2.5 rad) of tracking error. tau_max is the real
// motor limit, which is why the legs and waist are stiffer than G1's values.
std::array<float, H2X_NUM_MOTOR> Kp{
    150, 160, 160, 250, 90, 60,   // legs
    150, 160, 160, 250, 90, 60,   // legs
    300, 400, 1000,               // waist (yaw, roll, pitch)
    90, 55, 20, 55, 20, 10, 10,   // arms
    90, 55, 20, 55, 20, 10, 10,   // arms
    60, 60                        // head
};

// Damping for all H2X Joints: zeta ~ 0.1 with the measured joint inertia
std::array<float, H2X_NUM_MOTOR> Kd{
    2, 2, 2, 2, 0.1, 0.3,         // legs
    2, 2, 2, 2, 0.1, 0.3,         // legs
    10, 20, 50,                   // waist (yaw, roll, pitch)
    2, 1, 0.4, 1, 0.3, 0.3, 0.3,  // arms
    2, 1, 0.4, 1, 0.3, 0.3, 0.3,  // arms
    2, 2                          // head
};

static constexpr double kSeqSpeed = 1.0;

void PrintUsage() {
  std::cout
      << "Usage: h2x_dual_arm_example <network_interface> [options]\n"
      << "  <network_interface>  dds interface, e.g. eth0\n"
      << "\n"
      << "Drives all 31 joints over rt/lowcmd at 500 Hz (H2X layout).\n"
      << "Keep the robot suspended or otherwise protected.\n"
      << "\n"
      << "Stages:\n"
      << "  1  move all joints to the zero posture, 3 s\n"
      << "  2  replay motion.seq over the 14 arm joints 15~28, looping\n"
      << "     legs, waist 12~14 and head 29/30 hold the zero posture\n"
      << "\n"
      << "Options:\n"
      << "  -h, --help  show this help\n"
      << "\n"
      << "Environment:\n"
      << "  H2X_DUAL_ARM_BEHAVIOR_DIR  directory holding <behavior>.seq, it\n"
      << "                             overrides the build-time default below\n"
      << "                             " << H2X_DUAL_ARM_BEHAVIOR_DIR << "\n"
      << "\n"
      << "Ctrl+C stops the command loop.\n";
}

class H2XExample {
 private:
  double time_;
  double control_dt_;  // [2ms]
  double duration_;    // [3 s]
  PRorAB mode_;
  uint8_t mode_machine_;
  std::vector<std::vector<double>> frames_data_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;

  ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  ThreadPtr command_writer_ptr_, control_thread_ptr_;

 public:
  H2XExample()
      : time_(0.0),
        control_dt_(0.002),
        duration_(3.0),
        mode_(PR),
        mode_machine_(0) {

    loadBehaviorLibrary("motion");

    // create publisher
    lowcmd_publisher_.reset(
        new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();

    // create subscriber
    lowstate_subscriber_.reset(
        new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
            HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&H2XExample::LowStateHandler, this, std::placeholders::_1), 1);

    // create threads
    command_writer_ptr_ =
        CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000,
                                &H2XExample::LowCommandWriter, this);
    control_thread_ptr_ = CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, 2000, &H2XExample::Control, this);
  }

  void loadBehaviorLibrary(std::string behavior_name) {
    std::string resource_dir = BehaviorDir();
    YAML::Node motion = YAML::LoadFile(resource_dir + behavior_name + ".seq");

    std::string content = motion["components"][1]["content"].as<std::string>();
    int num_parts = motion["components"][1]["num_parts"].as<int>();
    std::cout << "BehaviorName: " << behavior_name + ".seq\n";
    std::cout << content << " with " << num_parts << "\n";

    auto frames = motion["components"][1]["frames"];

    for (const auto &frame : frames) {
      std::vector<double> frame_data;
      for (const auto &element : frame) {
        frame_data.push_back(element.as<double>());
      }
      frames_data_.push_back(frame_data);
    }

    std::cout << frames_data_.size() << " knots with " << frames_data_[0].size()
              << " DOF\n";
  }

  void ReportRPY() {
    const std::shared_ptr<const ImuState> imu_tmp_ptr =
        imu_state_buffer_.GetData();
    if (imu_tmp_ptr) {
      std::cout << "rpy: [" << imu_tmp_ptr->rpy.at(0) << ", "
                << imu_tmp_ptr->rpy.at(1) << ", " << imu_tmp_ptr->rpy.at(2)
                << "]" << std::endl;
    }
  }

  void LowStateHandler(const void *message) {
    unitree_hg::msg::dds_::LowState_ low_state =
        *(const unitree_hg::msg::dds_::LowState_ *)message;

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < H2X_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
    }
    motor_state_buffer_.SetData(ms_tmp);

    // get imu state
    ImuState imu_tmp;
    imu_tmp.omega = low_state.imu_state().gyroscope();
    imu_tmp.rpy = low_state.imu_state().rpy();
    imu_state_buffer_.SetData(imu_tmp);

    // update mode machine
    if (mode_machine_ != low_state.mode_machine()) {
      if (mode_machine_ == 0)
        std::cout << "H2X type: " << unsigned(low_state.mode_machine())
                  << std::endl;
      mode_machine_ = low_state.mode_machine();
    }
  }

  void LowCommandWriter() {
    unitree_hg::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = mode_;
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc =
        motor_command_buffer_.GetData();
    if (mc) {
      for (size_t i = 0; i < H2X_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
        dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
        dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
        dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
        dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
        dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
      }

      lowcmd_publisher_->Write(dds_low_command);
    }
  }

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

    if (ms) {
      time_ += control_dt_;
      if (time_ < duration_) {
        // [Stage 1]: set robot to zero posture
        for (int i = 0; i < H2X_NUM_MOTOR; ++i) {
          double ratio = std::clamp(time_ / duration_, 0.0, 1.0);

          double q_des = 0;
          motor_command_tmp.tau_ff.at(i) = 0.0;
          motor_command_tmp.q_target.at(i) =
              (q_des - ms->q.at(i)) * ratio + ms->q.at(i);
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = Kp[i];
          motor_command_tmp.kd.at(i) = Kd[i];
        }
      } else {
        // [Stage 2]: tracking the offline trajectory
        size_t frame_index = (size_t)(
            (time_ - duration_) * kSeqSpeed / control_dt_);
        if (frame_index >= frames_data_.size()) {
          frame_index = frames_data_.size() - 1;
          time_ = 0.0;  // RESET
        }

        if (frame_index % 100 == 0)
          std::cout << "Frame Index: " << frame_index << std::endl;

        for (int i = 0; i < H2X_NUM_MOTOR; ++i) {
          // The .seq file only carries the 14 arm joints; legs, waist and head
          // stay at the zero posture.
          const bool is_arm = (i >= LeftShoulderPitch) && (i <= RightWristYaw);
          const size_t index_in_frame = i - LeftShoulderPitch;
          motor_command_tmp.q_target.at(i) =
              is_arm ? frames_data_[frame_index][index_in_frame] : 0.0;
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.tau_ff.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = Kp[i];
          motor_command_tmp.kd.at(i) = Kd[i];
        }
      }

      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }

};

int main(int argc, char const *argv[]) {
  const std::string first_arg = argc > 1 ? argv[1] : "";
  if (first_arg == "-h" || first_arg == "--help") {
    PrintUsage();
    return 0;
  }
  if (argc < 2) {
    PrintUsage();
    return 0;
  }

  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "dds init failed: " << e.what() << std::endl;
    return 1;
  }

  H2XExample custom;

  while (true) sleep(10);

  return 0;
}
