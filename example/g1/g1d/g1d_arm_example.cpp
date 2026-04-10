#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <chrono>

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

const int G1D_NUM_MOTOR = 29;

// IMU状态结构体
struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
};

// 电机控制指令结构体
struct MotorCommand {
  std::array<float, G1D_NUM_MOTOR> q_target = {};
  std::array<float, G1D_NUM_MOTOR> dq_target = {};
  std::array<float, G1D_NUM_MOTOR> kp = {};
  std::array<float, G1D_NUM_MOTOR> kd = {};
  std::array<float, G1D_NUM_MOTOR> tau_ff = {};
};

// 电机状态结构体
struct MotorState {
  std::array<float, G1D_NUM_MOTOR> q = {};
  std::array<float, G1D_NUM_MOTOR> dq = {};
};

// Stiffness for all G1D Joints
std::array<float, G1D_NUM_MOTOR> Kp{
    0,  0,  0,  0,  0,  0,      // empty
    0,  0,  0,  0,  0,  0,      // empty
    60, 0,  40,                   // waist
    40, 40, 40, 40, 40, 40, 40,  // arms
    40, 40, 40, 40, 40, 40, 40   // arms
};

// Damping for all G1D Joints
std::array<float, G1D_NUM_MOTOR> Kd{
    0, 0, 0, 0, 0, 0,     // empty
    0, 0, 0, 0, 0, 0,     // empty
    1, 0, 1,              // waist
    1, 1, 1, 1, 1, 1, 1,  // arms
    1, 1, 1, 1, 1, 1, 1   // arms
};

uint8_t mode_machine_;

Gamepad gamepad_;
REMOTE_DATA_RX rx_;

DataBuffer<MotorState> motor_state_buffer_;
DataBuffer<MotorCommand> motor_command_buffer_;
DataBuffer<ImuState> imu_state_buffer_;

ThreadPtr command_writer_ptr_, control_thread_ptr_;

LowState_ g1d_low_state;
IMUState_ Torso_IMU;

bool A_PRESSED = false;
bool IMU_Get = false;

enum G1DJointIndex {
  LeftHipPitch = 0,      // NOTE INVALID for g1d
  LeftHipRoll = 1,       // NOTE INVALID for g1d
  LeftHipYaw = 2,        // NOTE INVALID for g1d
  LeftKnee = 3,          // NOTE INVALID for g1d
  LeftAnklePitch = 4,    // NOTE INVALID for g1d
  LeftAnkleB = 4,        // NOTE INVALID for g1d
  LeftAnkleRoll = 5,     // NOTE INVALID for g1d
  LeftAnkleA = 5,        // NOTE INVALID for g1d
  RightHipPitch = 6,     // NOTE INVALID for g1d
  RightHipRoll = 7,      // NOTE INVALID for g1d
  RightHipYaw = 8,       // NOTE INVALID for g1d
  RightKnee = 9,         // NOTE INVALID for g1d
  RightAnklePitch = 10,  // NOTE INVALID for g1d
  RightAnkleB = 10,      // NOTE INVALID for g1d
  RightAnkleRoll = 11,   // NOTE INVALID for g1d
  RightAnkleA = 11,      // NOTE INVALID for g1d
  WaistYaw = 12,
  WaistRoll = 13,        // NOTE INVALID for g1d
  WaistA = 13,           // NOTE INVALID for g1d
  WaistPitch = 14,
  WaistB = 14,           // NOTE INVALID for g1d
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
  RightWristYaw = 28
};

// CRC校验值计算
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

// 躯干IMU消息回调函数
void imuTorsoHandler(const void *message) {
  IMUState_ imu_torso = *(const IMUState_ *)message;
  Torso_IMU = imu_torso;
}

// 底层状态消息回调函数
void LowStateHandler(const void *message) {
  LowState_ low_state = *(const LowState_ *)message;
  g1d_low_state = low_state;
  if (low_state.crc() != Crc32Core((uint32_t *)&low_state, (sizeof(LowState_) >> 2) - 1)) {
    std::cout << "[ERROR] CRC Error" << std::endl;
    return;
  }

  // 获取关节电机数据
  MotorState ms_tmp;
  for (int i = 0; i < G1D_NUM_MOTOR; ++i) {
    ms_tmp.q.at(i) = low_state.motor_state()[i].q();
    ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
    if (low_state.motor_state()[i].motorstate() && i <= RightAnkleRoll)
      std::cout << "[ERROR] motor " << i << " with code " << low_state.motor_state()[i].motorstate() << "\n";
  }
  motor_state_buffer_.SetData(ms_tmp);

  // 获取IMU数据
  ImuState imu_tmp;
  imu_tmp.omega = low_state.imu_state().gyroscope();
  imu_tmp.rpy = low_state.imu_state().rpy();
  imu_state_buffer_.SetData(imu_tmp);
  IMU_Get = true;

  // 更新遥控器状态
  memcpy(rx_.buff, &low_state.wireless_remote()[0], 40);
  gamepad_.update(rx_.RF_RX);
  if(gamepad_.A.on_press) A_PRESSED = true;

  // 更新机型编号
  mode_machine_ = low_state.mode_machine();
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: g1d_arm_example network_interface" << std::endl;
    exit(0);
  }
  std::string networkInterface = argv[1];
  ChannelFactory::Instance()->Init(0, networkInterface);

  // Publisher
  ChannelPublisherPtr<LowCmd_> lowcmd_publisher(new ChannelPublisher<LowCmd_>(HG_CMD_TOPIC));
  lowcmd_publisher->InitChannel();

  // Subscriber
  ChannelSubscriberPtr<LowState_> lowstate_subscriber(new ChannelSubscriber<LowState_>(HG_STATE_TOPIC));
  lowstate_subscriber->InitChannel(LowStateHandler, 1);
  ChannelSubscriberPtr<IMUState_> imu_subscriber(new ChannelSubscriber<IMUState_>(HG_IMU_TORSO));
  imu_subscriber->InitChannel(imuTorsoHandler, 1);

  // 等待接收到第一帧IMU数据，打印躯干IMU状态信息
  while(!IMU_Get){
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "Torso IMU State: " << std::endl
            << "RPY = " 
            << Torso_IMU.rpy()[0] << ", " << Torso_IMU.rpy()[1] << ", " << Torso_IMU.rpy()[2]
            << std::endl
            << "Gyro = "
            << Torso_IMU.gyroscope()[0] << ", " << Torso_IMU.gyroscope()[1] << ", " << Torso_IMU.gyroscope()[2]
            << std::endl;

  // 等待遥控器A键按下
  std::cout << "Press button A on the remote control to continue!" << std::endl;
  while(true){
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    if(A_PRESSED){
      A_PRESSED = false;
      break;
    }
  }
  std::cout << "A Pressed!" << std::endl;

  // 手臂摆动
  LowCmd_ dds_low_command;
  dds_low_command.mode_pr() = static_cast<uint8_t>(0); //构造底层控制指令
  dds_low_command.mode_machine() = mode_machine_;
  for (size_t i = 0; i < G1D_NUM_MOTOR; i++) {
    dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
    dds_low_command.motor_cmd().at(i).tau() = 0;
    dds_low_command.motor_cmd().at(i).q() = motor_state_buffer_.GetData() -> q.at(i);
    dds_low_command.motor_cmd().at(i).dq() = 0;
    dds_low_command.motor_cmd().at(i).kp() = Kp.at(i);
    dds_low_command.motor_cmd().at(i).kd() = Kd.at(i);
  }

  double control_dt = 0.002;
  double max_P = M_PI * 30.0 / 180.0;
  double t = 0.0;

  std::cout << "Press button A on the remote control to exit the program!" << std::endl;
  while(true){
    double L_Shoulder_des = max_P * std::sin(2.0 * M_PI * t);
    double R_Shoulder_des = -max_P * std::sin(2.0 * M_PI * t);

    dds_low_command.motor_cmd().at(LeftShoulderPitch).q() = L_Shoulder_des;
    dds_low_command.motor_cmd().at(RightShoulderPitch).q() = R_Shoulder_des;

    dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command, (sizeof(dds_low_command) >> 2) - 1);
    lowcmd_publisher -> Write(dds_low_command); //发布底层控制指令

    t += control_dt;

    // 按下遥控器A键或键盘Ctrl+C退出程序
    if(A_PRESSED){
      A_PRESSED = false;
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  return 0;
}
