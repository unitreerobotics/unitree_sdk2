#include "unitree/dds_wrapper/robots/g1/g1.h"
#include "unitree/dds_wrapper/robots/r1/r1.h"

// High-level arm controller for the R1 robot.
// It subscribes to the robot's low-level state (joint feedback) and publishes
// arm commands over the "rt/arm_sdk" DDS topic, providing convenience helpers
// for enabling/disabling the arms and moving the joints to target positions.
class ArmController
{
public:
    ArmController()
    {
        // Subscribe to the low-level state so we can read the current joint positions.
        lowstate_ = std::make_shared<unitree::robot::g1::subscription::LowState>();
        std::cout << "Waiting for connection to lowstate..." << std::endl;
        lowstate_->wait_for_connection();
        // Create the arm-sdk publisher that sends joint commands to the robot.
        armsdk_ = std::make_unique<unitree::robot::r1::publisher::ArmSdk>();
        std::cout << "Connected to robot!" << std::endl;
    }


    // Take control of the arms: ramp the SDK weight up to 1.0 and initialize each
    // joint command to its current measured position so the arms hold still on start.
    void enable()
    {
        // Already enabled (weight near 1.0), nothing to do.
        if(armsdk_->weight() > 0.5) return;

        // weight == 1.0 hands full control of the arms to this SDK.
        armsdk_->weight(1.0);
        for(int i(0); i<armsdk_->JOINTS.size(); ++i)
        {
            int JOINT = static_cast<int>(armsdk_->JOINTS[i]);
            // Seed the target with the current position to avoid a sudden jump.
            armsdk_->msg_.motor_cmd().at(JOINT).q(lowstate_->msg_.motor_state().at(JOINT).q());
            // Position/velocity control gains for this joint.
            armsdk_->msg_.motor_cmd().at(JOINT).kp(KP[i]);
            armsdk_->msg_.motor_cmd().at(JOINT).kd(KD[i]);
            // No desired velocity or feed-forward torque.
            armsdk_->msg_.motor_cmd().at(JOINT).dq(0);
            armsdk_->msg_.motor_cmd().at(JOINT).tau(0);
        }
    }

    // Gradually release control of the arms by ramping the SDK weight from 1.0
    // down to 0.0 over `duration` seconds, then let the robot regain control.
    void release(float duration = 1.0f)
    {
        // Already released (weight near 0.0), nothing to do.
        if(armsdk_->weight() < 0.5) return;

        auto t0 = std::chrono::high_resolution_clock::now();
        while(true)
        {
            // Elapsed time since the release started.
            float elasped_time = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - t0).count();
            if(elasped_time > duration) break;
            // Linearly interpolate the weight from 1.0 down to 0.0.
            float _i = (duration - elasped_time) / duration;
            armsdk_->weight(std::clamp(_i, 0.f, 1.f));
            armsdk_->unlockAndPublish();
            // Publish at ~100 Hz.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Write the desired joint positions (radians) into the command message.
    // `q_des` must be ordered to match armsdk_->JOINTS.
    void set_q(Eigen::VectorXf q_des)
    {
        for(int i(0); i<armsdk_->JOINTS.size(); ++i)
        {
            int JOINT = static_cast<int>(armsdk_->JOINTS[i]);
            armsdk_->msg_.motor_cmd().at(JOINT).q(q_des[i]);
        }
    }

    // Move the joints from their current positions to `q_target` (radians) using
    // linear interpolation. `max_vel` (rad/s) caps the speed of the fastest joint,
    // which determines the overall motion duration.
    void movej(Eigen::VectorXf q_target, float max_vel = 1.0)
    {
        // Read the current joint positions from the low-level state feedback.
        Eigen::VectorXf q_current(armsdk_->JOINTS.size());
        for(int i(0); i<armsdk_->JOINTS.size(); ++i)
        {
            int JOINT = static_cast<int>(armsdk_->JOINTS[i]);
            q_current[i] = lowstate_->msg_.motor_state().at(JOINT).q();
        }

        // Duration is set by the joint with the largest travel divided by max_vel.
        float duration = (q_target - q_current).cwiseAbs().maxCoeff() / max_vel;
        for(float t(0); t < duration; t += 0.01)
        {
            // Interpolation ratio in [0, 1].
            float _i = t / duration;
            Eigen::VectorXf q_des = q_current + (q_target - q_current) * _i;
            set_q(q_des);
            armsdk_->unlockAndPublish();
            // Send commands at ~100 Hz.
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    std::shared_ptr<unitree::robot::g1::subscription::LowState> lowstate_;  // Low-level joint state feedback
    std::unique_ptr<unitree::robot::r1::publisher::ArmSdk> armsdk_;         // Arm command publisher

private:
    // Proportional (stiffness) gains, one per joint in armsdk_->JOINTS order.
    std::vector<float> KP = {
        50.0, 50.0, 40., 40., 30.,
        50.0, 50.0, 40., 40., 30.,
        50., 15., 15.
    };
    // Derivative (damping) gains, one per joint in armsdk_->JOINTS order.
    std::vector<float> KD = {
        2.0, 2.0, 2., 2., 2.,
        2.0, 2.0, 2., 2., 2.,
        3., 1., 1.
    };
};


int main(int argc, char const *argv[])
{
  // The network interface used for DDS communication must be provided.
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " networkInterface" << std::endl;
    exit(-1);
  }

  // Initialize the DDS channel factory on the given network interface.
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  auto arm = std::make_unique<ArmController>();

  // Take control of the arms.
  arm->enable();

  // Target joint positions (radians), ordered to match ArmSdk::JOINTS:
  // left arm (5), right arm (5), waist yaw, head pitch, head yaw.
  Eigen::VectorXf q_target(arm->armsdk_->JOINTS.size());
  q_target << 0.0, 1.57, 0.0, 1.57, 0.0,
              0.0, -1.57, 0.0, -1.57, 0.0,
              0.0, 0.0, 1.0;
  // Move to the target pose, then smoothly release control back to the robot.
  arm->movej(q_target, 1.0);
  arm->release(1.0);
}