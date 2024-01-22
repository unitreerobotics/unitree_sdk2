#pragma once

#include <array>
#include <stdint.h>

#include <unitree/idl/go2/LowCmd_.hpp>

constexpr int kNumMotors = 20;

struct MotorCommand {
  std::array<float, kNumMotors> q_ref = {};
  std::array<float, kNumMotors> dq_ref = {};
  std::array<float, kNumMotors> kp = {};
  std::array<float, kNumMotors> kd = {};
  std::array<float, kNumMotors> tau_ff = {};
};

struct MotorState {
  std::array<float, kNumMotors> q = {};
  std::array<float, kNumMotors> dq = {};
};

enum JointIndex {
  // Right leg
  kRightHipYaw = 8,
  kRightHipRoll = 0,
  kRightHipPitch = 1,
  kRightKnee = 2,
  kRightAnkle = 11,
  // Left leg
  kLeftHipYaw = 7,
  kLeftHipRoll = 3,
  kLeftHipPitch = 4,
  kLeftKnee = 5,
  kLeftAnkle = 10,

  kWaistYaw = 6,

  kNotUsedJoint = 9,

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

#pragma pack(1)

struct PackOneBmsCmd {
  uint8_t off = 0x00;
  std::array<uint8_t, 3> reserve;
};

// motor control
struct PackOneMotorCommand {
  // desired working mode
  uint8_t mode = 0x01;
  // desired angle (unit: radian)
  float q = 0;
  // desired velocity (unit: radian/second)
  float dq = 0;
  // desired output torque (unit: N.m)
  float tau = 0;
  // desired position stiffness (unit: N.m/rad )
  float kp = 0;
  // desired velocity stiffness (unit: N.m/(rad/s) )
  float kd = 0;
  std::array<uint32_t, 3> reserve = {0};
};

// low level control
struct PackOneLowCommand {
  std::array<uint8_t, 2> head = {0xFE, 0xEF};
  uint8_t level_flag = 0xFF;
  uint8_t frame_reserve = 0;

  std::array<uint32_t, 2> sn;
  std::array<uint32_t, 2> version;
  uint16_t band_width = 0;
  std::array<PackOneMotorCommand, 20> motor_command;
  PackOneBmsCmd bms;
  std::array<uint8_t, 40> wireless_remote;
  std::array<uint8_t, 12> led;
  std::array<uint8_t, 2> fan;
  uint8_t gpio = 0;
  uint32_t reserve;

  uint32_t crc;
};

#pragma pack()

uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
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
      if (data & xbit)
        CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
};

void LowCmd2DDS(const PackOneLowCommand &pack_one_low_command,
                unitree_go::msg::dds_::LowCmd_ &dds_low_command) {
  memcpy(&dds_low_command.head()[0], &pack_one_low_command.head[0], 2);
  dds_low_command.level_flag() = pack_one_low_command.level_flag;
  dds_low_command.frame_reserve() = pack_one_low_command.frame_reserve;
  memcpy(&dds_low_command.sn()[0], &pack_one_low_command.sn[0], 8);
  memcpy(&dds_low_command.version()[0], &pack_one_low_command.version[0], 8);
  dds_low_command.bandwidth() = pack_one_low_command.band_width;

  // must use 20
  for (int i = 0; i < 20; ++i) {
    dds_low_command.motor_cmd()[i].mode() =
        pack_one_low_command.motor_command[i].mode;
    dds_low_command.motor_cmd()[i].q() =
        pack_one_low_command.motor_command[i].q;
    dds_low_command.motor_cmd()[i].dq() =
        pack_one_low_command.motor_command[i].dq;
    dds_low_command.motor_cmd()[i].tau() =
        pack_one_low_command.motor_command[i].tau;
    dds_low_command.motor_cmd()[i].kp() =
        pack_one_low_command.motor_command[i].kp;
    dds_low_command.motor_cmd()[i].kd() =
        pack_one_low_command.motor_command[i].kd;

    memcpy(&dds_low_command.motor_cmd()[i].reserve()[0],
           &pack_one_low_command.motor_command[i].reserve[0], 12);
  }

  dds_low_command.bms_cmd().off() = pack_one_low_command.bms.off;
  memcpy(&dds_low_command.bms_cmd().reserve()[0],
         &pack_one_low_command.bms.reserve[0], 3);

  memcpy(&dds_low_command.wireless_remote()[0],
         &pack_one_low_command.wireless_remote[0], 40);

  memcpy(&dds_low_command.led()[0], &pack_one_low_command.led[0], 12);
  memcpy(&dds_low_command.fan()[0], &pack_one_low_command.fan[0], 2);
  dds_low_command.gpio() = pack_one_low_command.gpio;
  dds_low_command.reserve() = pack_one_low_command.reserve;
  dds_low_command.crc() = pack_one_low_command.crc;
};
