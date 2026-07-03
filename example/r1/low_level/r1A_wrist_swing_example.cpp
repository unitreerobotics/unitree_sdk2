#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <algorithm>

#include "gamepad.hpp"

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

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

const int R1Z_NUM_MOTOR = 12;
struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
};
struct MotorCommand {
  std::array<float, R1Z_NUM_MOTOR> q_target = {};
  std::array<float, R1Z_NUM_MOTOR> dq_target = {};
  std::array<float, R1Z_NUM_MOTOR> kp = {};
  std::array<float, R1Z_NUM_MOTOR> kd = {};
  std::array<float, R1Z_NUM_MOTOR> tau_ff = {};
};
struct MotorState {
  std::array<float, R1Z_NUM_MOTOR> q = {};
  std::array<float, R1Z_NUM_MOTOR> dq = {};
};

// Stiffness for all R1Z joints
std::array<float, R1Z_NUM_MOTOR> Kp{
    100, 100, 100, 100, 50,  // left arm
    100, 100, 100, 100, 50,  // right arm
    50, 10                   // head
};

// Damping for all R1Z joints
std::array<float, R1Z_NUM_MOTOR> Kd{
    2, 2, 2, 2, 2,  // left arm
    2, 2, 2, 2, 2,  // right arm
    2, 0.1          // head
};

enum R1JointIndex {
  LeftShoulderPitch = 0,
  LeftShoulderRoll = 1,
  LeftShoulderYaw = 2,
  LeftElbow = 3,
  LeftWristRoll = 4,
  RightShoulderPitch = 5,
  RightShoulderRoll = 6,
  RightShoulderYaw = 7,
  RightElbow = 8,
  RightWristRoll = 9,
  HEAD_PITCH = 10,
  HEAD_YAW = 11,
};

std::array<int, R1Z_NUM_MOTOR> joint_idx_in_idl{
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
    29, 30
};

inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else
        CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
};

class R1Example {
 private:
  double time_;
  double control_dt_;  // [2ms]
  double duration_;    // [2 s]
  int counter_;
  uint8_t mode_pr_;
  uint8_t mode_machine_;

  Gamepad gamepad_;
  REMOTE_DATA_RX rx_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;

  ChannelPublisherPtr<LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<LowState_> lowstate_subscriber_;
  ChannelSubscriberPtr<IMUState_> imutorso_subscriber_;
  ThreadPtr command_writer_ptr_, control_thread_ptr_;

  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> msc_;

 public:
  R1Example(std::string networkInterface)
      : time_(0.0),
        control_dt_(0.002),
        duration_(2.0),
        counter_(0),
        mode_pr_(0),
        mode_machine_(0) {
    ChannelFactory::Instance()->Init(0, networkInterface);

    // try to shutdown motion control-related service
    msc_ = std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
    msc_->SetTimeout(5.0f);
    msc_->Init();
    std::string form, name;
    while (msc_->CheckMode(form, name), !name.empty()) {
      if (msc_->ReleaseMode())
        std::cout << "Failed to switch to Release Mode\n";
      sleep(5);
    }

    // create publisher
    lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();
    // create subscriber
    lowstate_subscriber_.reset(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(std::bind(&R1Example::LowStateHandler, this, std::placeholders::_1), 1);
    imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber_->InitChannel(std::bind(&R1Example::imuTorsoHandler, this, std::placeholders::_1), 1);
    // create threads
    command_writer_ptr_ = CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000, &R1Example::LowCommandWriter, this);
    control_thread_ptr_ = CreateRecurrentThreadEx("control", UT_CPU_ID_NONE, 2000, &R1Example::Control, this);
  }

  void imuTorsoHandler(const void *message) {
    IMUState_ imu_torso = *(const IMUState_ *)message;
    auto &rpy = imu_torso.rpy();
    if (counter_ % 500 == 0)
      printf("IMU.torso.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
  }

  void LowStateHandler(const void *message) {
    LowState_ low_state = *(const LowState_ *)message;
    if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) {
      std::cout << "[ERROR] CRC Error" << std::endl;
      return;
    }

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < R1Z_NUM_MOTOR; ++i) {
      int idl_index = joint_idx_in_idl[i];
      ms_tmp.q.at(i) = low_state.motor_state()[idl_index].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[idl_index].dq();
      if (low_state.motor_state()[idl_index].motorstate())
        std::cout << "[ERROR] motor " << idl_index << " with code " << low_state.motor_state()[idl_index].motorstate() << "\n";
    }
    motor_state_buffer_.SetData(ms_tmp);

    // get imu state
    ImuState imu_tmp;
    imu_tmp.omega = low_state.imu_state().gyroscope();
    imu_tmp.rpy = low_state.imu_state().rpy();
    imu_state_buffer_.SetData(imu_tmp);

    // update gamepad
    memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
    gamepad_.update(rx_.RF_RX);

    // update mode machine
    if (mode_machine_ != low_state.mode_machine()) {
      if (mode_machine_ == 0) std::cout << "R1 type: " << unsigned(low_state.mode_machine()) << std::endl;
      mode_machine_ = low_state.mode_machine();
    }

    // report robot status every second
    if (++counter_ % 500 == 0) {
      counter_ = 0;
      // IMU
      auto &rpy = low_state.imu_state().rpy();
      printf("IMU.pelvis.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);

      // RC
      printf("gamepad_.A.pressed: %d\n", static_cast<int>(gamepad_.A.pressed));
      printf("gamepad_.B.pressed: %d\n", static_cast<int>(gamepad_.B.pressed));
      printf("gamepad_.X.pressed: %d\n", static_cast<int>(gamepad_.X.pressed));
      printf("gamepad_.Y.pressed: %d\n", static_cast<int>(gamepad_.Y.pressed));

      // Motor
      auto &ms = low_state.motor_state();
      printf("All %d Motors:", R1Z_NUM_MOTOR);
      printf("\nmode: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%u,", ms[joint_idx_in_idl[i]].mode());
      printf("\npos: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%.2f,", ms[joint_idx_in_idl[i]].q());
      printf("\nvel: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%.2f,", ms[joint_idx_in_idl[i]].dq());
      printf("\ntau_est: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%.2f,", ms[joint_idx_in_idl[i]].tau_est());
      printf("\ntemperature: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%d,%d;", ms[joint_idx_in_idl[i]].temperature()[0], ms[joint_idx_in_idl[i]].temperature()[1]);
      printf("\nvol: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%.2f,", ms[joint_idx_in_idl[i]].vol());
      printf("\nsensor: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%u,%u;", ms[joint_idx_in_idl[i]].sensor()[0], ms[joint_idx_in_idl[i]].sensor()[1]);
      printf("\nmotorstate: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%u,", ms[joint_idx_in_idl[i]].motorstate());
      printf("\nreserve: ");
      for (int i = 0; i < R1Z_NUM_MOTOR; ++i) printf("%u,%u,%u,%u;", ms[joint_idx_in_idl[i]].reserve()[0], ms[joint_idx_in_idl[i]].reserve()[1], ms[joint_idx_in_idl[i]].reserve()[2], ms[joint_idx_in_idl[i]].reserve()[3]);
      printf("\n");
    }
  }

  void LowCommandWriter() {
    LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = mode_pr_;
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
    if (mc) {
      for (int i = 0; i < R1Z_NUM_MOTOR; i++) {
        int idl_index = joint_idx_in_idl[i];
        dds_low_command.motor_cmd().at(idl_index).mode() = 1;  // 1:Enable, 0:Disable
        dds_low_command.motor_cmd().at(idl_index).tau() = mc->tau_ff.at(i);
        dds_low_command.motor_cmd().at(idl_index).q() = mc->q_target.at(i);
        dds_low_command.motor_cmd().at(idl_index).dq() = mc->dq_target.at(i);
        dds_low_command.motor_cmd().at(idl_index).kp() = mc->kp.at(i);
        dds_low_command.motor_cmd().at(idl_index).kd() = mc->kd.at(i);
      }

      dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
      lowcmd_publisher_->Write(dds_low_command);
    }
  }

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

    for (int i = 0; i < R1Z_NUM_MOTOR; ++i) {
      motor_command_tmp.tau_ff.at(i) = 0.0;
      motor_command_tmp.q_target.at(i) = 0.0;
      motor_command_tmp.dq_target.at(i) = 0.0;
      motor_command_tmp.kp.at(i) = Kp[i];
      motor_command_tmp.kd.at(i) = Kd[i];
    }

    if (ms) {
      time_ += control_dt_;
      if (time_ < duration_) {
        // [Stage 1]: set r1z arms and head to zero posture
        double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
        for (int i = 0; i < R1Z_NUM_MOTOR; ++i) {
          motor_command_tmp.q_target.at(i) = (1.0 - ratio) * ms->q.at(i);
        }
      } else {
        const double t = std::fmod(time_ - duration_, duration_ * 2.0);
        mode_pr_ = 0;

        if (t < duration_) {
          // [Stage 2]: swing wrist roll joints for 2 s
          double max_wrist_roll = M_PI * 30.0 / 180.0;
          double wrist_roll_des = max_wrist_roll * std::sin(2.0 * M_PI * t / duration_);

          motor_command_tmp.q_target.at(LeftWristRoll) = wrist_roll_des;
          motor_command_tmp.q_target.at(RightWristRoll) = -wrist_roll_des;
        } else {
          // [Stage 3]: swing head pitch and yaw joints for 2 s
          double head_t = t - duration_;
          double max_head_yaw = M_PI * 30.0 / 180.0;
          motor_command_tmp.q_target.at(HEAD_YAW) = max_head_yaw * std::sin(2.0 * M_PI * head_t / duration_);
        }
      }

      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: r1A_wrist_swing_example network_interface" << std::endl;
    exit(0);
  }
  std::string networkInterface = argv[1];
  R1Example custom(networkInterface);
  while (true) sleep(10);
  return 0;
}
