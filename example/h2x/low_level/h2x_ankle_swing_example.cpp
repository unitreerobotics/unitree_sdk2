#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>

#include "gamepad.hpp"

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>

// H2X low level : 31 motors, ankle pitch 4 / roll 5, waist 12~14, head 29/30.
static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_IMU_TORSO = "rt/secondary_imu";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

template <typename T> class DataBuffer {
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

const int H2X_NUM_MOTOR = 31;
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

enum class Mode {
  PR = 0, // Series Control for Ptich/Roll Joints
  AB = 1  // Parallel Control for A/B Joints
};

enum H2XJointIndex {
  LeftHipPitch = 0,
  LeftHipRoll = 1,
  LeftHipYaw = 2,
  LeftKnee = 3,
  LeftAnklePitch = 4,
  LeftAnkleB = 4,
  LeftAnkleRoll = 5,
  LeftAnkleA = 5,
  RightHipPitch = 6,
  RightHipRoll = 7,
  RightHipYaw = 8,
  RightKnee = 9,
  RightAnklePitch = 10,
  RightAnkleB = 10,
  RightAnkleRoll = 11,
  RightAnkleA = 11,
  WaistYaw = 12,
  WaistRoll = 13,
  WaistA = 13,
  WaistPitch = 14,
  WaistB = 14,
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
    150, 180, 200,                // waist
    90, 55, 20, 55, 20, 10, 10,   // arms
    90, 55, 20, 55, 20, 10, 10,   // arms
    30, 30                        // head
};

// Damping for all H2X Joints: zeta ~ 0.1 with the measured joint inertia
std::array<float, H2X_NUM_MOTOR> Kd{
    2, 2, 2, 2, 0.1, 0.3,     // legs
    2, 2, 2, 2, 0.1, 0.3,     // legs
    2.5, 5, 5,                // waist
    2, 1, 0.4, 1, 0.3, 0.3, 0.3,  // arms
    2, 1, 0.4, 1, 0.3, 0.3, 0.3,  // arms
    1, 1                          // head
};

class H2XExample {
private:
  double time_;
  double control_dt_; // [2ms]
  double duration_;   // [3 s]
  int counter_;
  Mode mode_pr_;
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


public:
  H2XExample(std::string networkInterface)
      : time_(0.0), control_dt_(0.002), duration_(3.0), counter_(0),
        mode_pr_(Mode::PR), mode_machine_(0) {

    // create publisher
    lowcmd_publisher_.reset(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();
    // create subscriber
    lowstate_subscriber_.reset(
        new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&H2XExample::LowStateHandler, this, std::placeholders::_1), 1);
    imutorso_subscriber_.reset(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
    imutorso_subscriber_->InitChannel(
        std::bind(&H2XExample::imuTorsoHandler, this, std::placeholders::_1), 1);
    // create threads
    command_writer_ptr_ =
        CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000,
                                &H2XExample::LowCommandWriter, this);
    control_thread_ptr_ = CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, 2000, &H2XExample::Control, this);
  }

  void imuTorsoHandler(const void *message) {
    IMUState_ imu_torso = *(const IMUState_ *)message;
    auto &rpy = imu_torso.rpy();
    if (counter_ % 500 == 0)
      printf("IMU.torso.rpy: %.2f %.2f %.2f\n", rpy[0], rpy[1], rpy[2]);
  }

  void LowStateHandler(const void *message) {
    LowState_ low_state = *(const LowState_ *)message;

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < H2X_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
      if (low_state.motor_state()[i].motorstate() && i <= RightAnkleRoll)
        std::cout << "[ERROR] motor " << i << " with code "
                  << low_state.motor_state()[i].motorstate() << "\n";
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
      if (mode_machine_ == 0)
        std::cout << "H2X type: " << unsigned(low_state.mode_machine())
                  << std::endl;
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
      printf("All %d Motors:", H2X_NUM_MOTOR);
      printf("\nmode: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%u,", ms[i].mode());
      printf("\npos: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%.2f,", ms[i].q());
      printf("\nvel: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%.2f,", ms[i].dq());
      printf("\ntau_est: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%.2f,", ms[i].tau_est());
      printf("\ntemperature: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%d,%d;", ms[i].temperature()[0], ms[i].temperature()[1]);
      printf("\nvol: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%.2f,", ms[i].vol());
      printf("\nsensor: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%u,%u;", ms[i].sensor()[0], ms[i].sensor()[1]);
      printf("\nmotorstate: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%u,", ms[i].motorstate());
      printf("\nreserve: ");
      for (int i = 0; i < H2X_NUM_MOTOR; ++i)
        printf("%u,%u,%u,%u;", ms[i].reserve()[0], ms[i].reserve()[1],
               ms[i].reserve()[2], ms[i].reserve()[3]);
      printf("\n");
    }
  }

  void LowCommandWriter() {
    LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = static_cast<uint8_t>(mode_pr_);
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc =
        motor_command_buffer_.GetData();
    if (mc) {
      for (size_t i = 0; i < H2X_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1; // 1:Enable, 0:Disable
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

    for (int i = 0; i < H2X_NUM_MOTOR; ++i) {
      motor_command_tmp.tau_ff.at(i) = 0.0;
      motor_command_tmp.q_target.at(i) = 0.0;
      motor_command_tmp.dq_target.at(i) = 0.0;
      motor_command_tmp.kp.at(i) = Kp[i];
      motor_command_tmp.kd.at(i) = Kd[i];
    }

    if (ms) {
      time_ += control_dt_;
      if (time_ < duration_) {
        // [Stage 1]: set robot to zero posture
        for (int i = 0; i < H2X_NUM_MOTOR; ++i) {
          double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
          motor_command_tmp.q_target.at(i) = (1.0 - ratio) * ms->q.at(i);
        }
      } else if (time_ < duration_ * 2) {
        // [Stage 2]: swing ankle using PR mode
        mode_pr_ = Mode::PR;
        // Real ankle range is pitch -50..+30 deg, roll +-15 deg.
        double max_P = M_PI * 25.0 / 180.0;
        double max_R = M_PI * 10.0 / 180.0;
        double t = time_ - duration_;
        double L_P_des = max_P * std::sin(2.0 * M_PI * t);
        double L_R_des = max_R * std::sin(2.0 * M_PI * t);
        double R_P_des = max_P * std::sin(2.0 * M_PI * t);
        double R_R_des = -max_R * std::sin(2.0 * M_PI * t);

        motor_command_tmp.q_target.at(LeftAnklePitch) = L_P_des;
        motor_command_tmp.q_target.at(LeftAnkleRoll) = L_R_des;
        motor_command_tmp.q_target.at(RightAnklePitch) = R_P_des;
        motor_command_tmp.q_target.at(RightAnkleRoll) = R_R_des;
      } else {
        // [Stage 3]: swing ankle using AB mode
        mode_pr_ = Mode::AB;
        // A/B are actuator coordinates, the model only gives pitch/roll
        // ranges, so keep the parallel-mode swing a bit smaller.
        double max_A = 0.2418;
        double max_B = M_PI * 8.0 / 180.0;
        double t = time_ - duration_ * 2;
        double L_A_des = +max_A * std::sin(M_PI * t);
        double L_B_des = +max_B * std::sin(M_PI * t + M_PI);
        double R_A_des = -max_A * std::sin(M_PI * t);
        double R_B_des = -max_B * std::sin(M_PI * t + M_PI);

        motor_command_tmp.q_target.at(LeftAnkleA) = L_A_des;
        motor_command_tmp.q_target.at(LeftAnkleB) = L_B_des;
        motor_command_tmp.q_target.at(RightAnkleA) = R_A_des;
        motor_command_tmp.q_target.at(RightAnkleB) = R_B_des;
      }

      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }
};

void PrintUsage() {
  std::cout
      << "Usage: h2x_ankle_swing_example <network_interface> [options]\n"
      << "  <network_interface>  dds interface, e.g. eth0\n"
      << "\n"
      << "Drives all 31 joints over rt/lowcmd at 500 Hz (H2X layout):\n"
      << "  knee 3/9, ankle pitch 4/10, ankle roll 5/11, waist 12~14,\n"
      << "  arms 15~28, head 29/30.\n"
      << "Keep the robot suspended or otherwise protected.\n"
      << "\n"
      << "Stages, 3 s each and looping:\n"
      << "  1  move all joints to the zero posture\n"
      << "  2  PR mode: ankle pitch +-25 deg, ankle roll +-10 deg\n"
      << "  3  AB mode: ankle A +-20 deg, ankle B +-8 deg\n"
      << "\n"
      << "Options:\n"
      << "  -h, --help  show this help\n"
      << "\n"
      << "Ctrl+C ends the process.\n";
}


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
  std::string networkInterface = argv[1];
  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface);
  } catch (const std::exception& e) {
    std::cerr << "dds init failed: " << e.what() << std::endl;
    return 1;
  }
  H2XExample custom(networkInterface);
  while (true)
    sleep(10);
  return 0;
}

