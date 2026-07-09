#pragma once

#include <eigen3/Eigen/Dense>
#include <unitree/dds_wrapper/common/Publisher.h>
#include <unitree/idl/hg/LowCmd_.hpp>
#include "defines.h"

namespace unitree
{
namespace robot
{
namespace r1
{
namespace publisher
{

class ArmSdk : public RealTimePublisher<unitree_hg::msg::dds_::LowCmd_>
{
public:
    ArmSdk(std::string topic = "rt/arm_sdk") : RealTimePublisher<MsgType>(topic) 
    {
    }

    void weight(float coe) { msg_.mode_pr() = std::clamp(int(coe * 100.0f), 0, 100); }

    float weight() const { return float(msg_.mode_pr()) / 100.0; }

    static constexpr std::array<JointIndex, 13> JOINTS = {
        JointIndex::LeftShoulderPitch, 
        JointIndex::LeftShoulderRoll,
        JointIndex::LeftShoulderYaw,   
        JointIndex::LeftElbow,
        JointIndex::LeftWristRoll,     
        JointIndex::RightShoulderPitch,
        JointIndex::RightShoulderRoll, 
        JointIndex::RightShoulderYaw,
        JointIndex::RightElbow,        
        JointIndex::RightWristRoll,
        JointIndex::WaistYaw,          
        JointIndex::HeadPitch,
        JointIndex::HeadYaw,
    };
};

}
}
}
}