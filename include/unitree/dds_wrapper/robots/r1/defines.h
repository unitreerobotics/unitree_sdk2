#pragma once

#include <array>

namespace unitree
{
namespace robot
{
namespace r1
{

enum class JointIndex
{
    // left leg
    LeftHipPitch = 0,
    LeftHipRoll,
    LeftHipYaw,
    LeftKnee,
    LeftAnklePitch,
    LeftAnkleRoll,

    // right leg
    RightHipPitch,
    RightHipRoll,
    RightHipYaw,
    RightKnee,
    RightAnklePitch,
    RightAnkleRoll,

    WaistRoll,
    WaistYaw,
    // no pitch

    // left arm
    LeftShoulderPitch = 15,
    LeftShoulderRoll,
    LeftShoulderYaw,
    LeftElbow,
    LeftWristRoll,

    // right arm
    RightShoulderPitch = 22,
    RightShoulderRoll,
    RightShoulderYaw,
    RightElbow,
    RightWristRoll,

    // head
    HeadPitch = 29,
    HeadYaw,
};


} // namespace r1
} // namespace robot
} // namespace unitree