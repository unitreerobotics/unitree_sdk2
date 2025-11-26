#ifndef __UT_ROBOT_G1_AGV_CLIENT_HPP__
#define __UT_ROBOT_G1_AGV_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>
#include "g1_agv_api.hpp"

namespace unitree {
namespace robot {
namespace g1 {


class AgvClient : public Client {
public:
    AgvClient() : Client(AGV_SERVICE_NAME, false) {}
    ~AgvClient() {}

    /*Init*/
    void Init() {
        SetApiVersion(AGV_API_VERSION);
        UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AGV_MOVE);
        UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AGV_HEIGHT_ADJUST);
    };

    /*API Call*/
    /**
    * @brief Linear and angular velocity control interface
    * 
    * Mechanical Parameters:
    *   - Max Linear Velocity: 1.5 m/s
    *   - Max Angular Velocity: 0.6 rad/s
    * 
    * Control Mapping:
    *   - vx: linear velocity in x direction
    *       Range: [-1.5, 1.5] m/s
    *       Positive value: move forward
    *       Negative value: move backward
    *       Zero: stop linear motion
    *   - vy: lateral velocity in y direction (not supported for AGV, ignored)
    *   - vyaw: angular velocity around z axis (rotation)
    *       Range: [-0.6, 0.6] rad/s
    *       Positive value: rotate counter-clockwise
    *       Negative value: rotate clockwise
    *       Zero: stop rotation
    * 
    * @param vx Linear velocity in x direction, unit: m/s, range: [-1.5, 1.5]
    * @param vy Lateral velocity in y direction (not used for AGV, pass 0.0)
    * @param vyaw Angular velocity around z axis, unit: rad/s, range: [-0.6, 0.6]
    * @return 0 on success, -1 on failure (not initialized or communication error)
    * 
    * @note Call Init() before using this function
    * @note This function is non-blocking and returns immediately after sending the command
    * @example
    *   ac.Move(0.5f, 0.0f, 0.3f);  // Move forward at 0.5 m/s and rotate at 0.3 rad/s
    */
    int32_t Move(float vx, float vy, float vyaw) {
        MoveParameter param;

        param.vx = vx;
        param.vy = vy;
        param.vyaw = vyaw;

        std::string parameter = common::ToJsonString(param);
        std::string data;

        return Call(ROBOT_API_ID_AGV_MOVE, parameter, data);
    }

    /**
    * @brief Height column velocity control interface
    * 
    * Mechanical Parameters:
    *   - Max Column Velocity: 76.5 mm/s (0.0765 m/s)
    * 
    * Normalized Linear Mapping (vz value → column velocity):
    *   The input vz is normalized [-1.0, +1.0] and linearly mapped to column velocity.
    *   Formula: column_velocity(mm/s) = vz × 76.5   
    *       vz = +1.0  →  +76.5 mm/s  (max rise)
    *       vz =  0.0  →  0 mm/s      (stop)
    *       vz = -1.0  →  -76.5 mm/s  (max descent)
    * 
     * Linear Mapping Benefits:
    *   - Intuitive control: proportional input produces proportional output
    *   - Predictable behavior: 0.5 input always gives exactly 50% of max velocity
    *   - Hardware independent: easy to adapt when motor parameters change
    *   - Simple conversion: column_velocity = normalized_input × max_velocity_constant
    
    * @param vz Height velocity command (range: [-1.0 , +1.0])
    * @return 0 on success, -1 on failure
    * 
    * @note Only z component is used, transmitted via DDS protocol to motor controller
    */
    int32_t HeightAdjust(float vz) {
        std::string parameter, data;

        go2::JsonizeDataFloat json;
        json.data = vz;

        parameter = common::ToJsonString(json);

        return Call(ROBOT_API_ID_AGV_HEIGHT_ADJUST, parameter, data);
    }
};
}  // namespace g1

}  // namespace robot
}  // namespace unitree

#endif // __UT_ROBOT_G1_AGV_CLIENT_HPP__
