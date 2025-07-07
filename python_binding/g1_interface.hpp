#pragma once

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <cmath>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>

#include "unitree/common/thread/thread.hpp"

using namespace unitree::common;
using namespace unitree::robot;

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";
static const std::string TOPIC_JOYSTICK = "rt/wirelesscontroller";

const int G1_NUM_MOTOR = 29;

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

// Python-exposed data structures
struct PyImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
  std::array<float, 4> quat = {};
  std::array<float, 3> accel = {};
};

struct PyMotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
  std::array<float, G1_NUM_MOTOR> tau_est = {};
  std::array<int, G1_NUM_MOTOR> temperature = {};
  std::array<float, G1_NUM_MOTOR> voltage = {};
};

struct PyMotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

struct PyWirelessController {
  std::array<float, 2> left_stick = {};   // [x, y]
  std::array<float, 2> right_stick = {};  // [x, y]
  bool A = false;
  bool B = false;
  bool X = false;
  bool Y = false;
  bool L1 = false;
  bool L2 = false;
  bool R1 = false;
  bool R2 = false;
};

struct PyLowState {
  PyImuState imu;
  PyMotorState motor;
  uint8_t mode_machine = 0;
};

enum class PyControlMode {
  PR = 0,  // Pitch/Roll mode
  AB = 1   // A/B mode
};

// Internal data structures (same as in the original files)
struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
  std::array<float, 4> quat = {};
  std::array<float, 3> accel = {};
};

struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
  std::array<float, G1_NUM_MOTOR> tau_est = {};
  std::array<int, G1_NUM_MOTOR> temperature = {};
  std::array<float, G1_NUM_MOTOR> voltage = {};
};

// CRC function
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
}

class G1Interface {
 private:
  PyControlMode mode_;
  uint8_t mode_machine_;
  
  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;
  DataBuffer<PyWirelessController> wireless_controller_buffer_;

  ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  ChannelSubscriberPtr<unitree_go::msg::dds_::WirelessController_> wireless_subscriber_;
  
  ThreadPtr command_writer_ptr_;
  std::mutex wireless_mutex_;

  // Default gains
  std::array<float, G1_NUM_MOTOR> default_kp_;
  std::array<float, G1_NUM_MOTOR> default_kd_;

  void InitDefaultGains();
  void LowStateHandler(const void *message);
  void WirelessControllerHandler(const void *message);
  void LowCommandWriter();
  
  // Convert internal structures to Python structures
  PyLowState ConvertToPyLowState();
  MotorCommand ConvertFromPyMotorCommand(const PyMotorCommand& py_cmd);

 public:
  G1Interface(const std::string& networkInterface);
  ~G1Interface();
  
  // Python interface methods
  PyLowState ReadLowState();
  PyWirelessController ReadWirelessController();
  void WriteLowCommand(const PyMotorCommand& command);
  void SetControlMode(PyControlMode mode);
  PyControlMode GetControlMode() const;
  
  // Utility methods
  PyMotorCommand CreateZeroCommand();
  std::array<float, G1_NUM_MOTOR> GetDefaultKp() const;
  std::array<float, G1_NUM_MOTOR> GetDefaultKd() const;
}; 