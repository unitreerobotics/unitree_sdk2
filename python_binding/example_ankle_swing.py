#!/usr/bin/env python3
"""
G1 Ankle Swing Example

This example demonstrates controlling the G1 robot's ankle joints
using the Python interface with PR (Pitch/Roll) mode.

Based on the original C++ ankle swing example.
"""

import sys
import time
import math
import signal
from typing import Optional

sys.path.append("/home/unitree/haoyang_deploy/unitree_sdk2/build/lib")

try:
    import g1_interface
except ImportError:
    print("Error: g1_interface module not found!")
    print("Please build the module first using: ./build.sh")
    print("Or make sure the compiled .so file is in your Python path")
    sys.exit(1)

class G1AnkleSwingController:
    """G1 Ankle swing controller using Python interface"""
    
    def __init__(self, network_interface: str):
        """
        Initialize the controller
        
        Args:
            network_interface: Network interface name (e.g., "eth0")
        """
        print(f"Initializing G1Interface with network interface: {network_interface}")
        
        self.robot = g1_interface.G1Interface(network_interface)
        self.running = True
        
        # Control parameters
        self.control_dt = 0.002  # 2ms control loop, 500Hz
        self.duration_stage = 3.0  # 3 seconds per stage
        self.current_time = 0.0
        self.stage = 0  # 0: init to zero, 1: PR ankle swing, 2: finished
        
        # Ankle swing parameters
        self.max_pitch = math.radians(20.0)  # 30 degrees
        self.max_roll = math.radians(10.0)   # 10 degrees
        
        # Set control mode to PR (Pitch/Roll)
        self.robot.set_control_mode(g1_interface.ControlMode.PR)
        control_mode = self.robot.get_control_mode()
        print(f"Control mode set to: PR" if control_mode == g1_interface.ControlMode.PR else f"AB")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
        
    def read_robot_state(self) -> g1_interface.LowState:
        """Read current robot state"""
        return self.robot.read_low_state()
        
    def read_controller_input(self) -> g1_interface.WirelessController:
        """Read wireless controller input"""
        return self.robot.read_wireless_controller()
        
    def create_zero_position_command(self, current_state: g1_interface.LowState) -> g1_interface.MotorCommand:
        """
        Create command to move robot to zero position
        
        Args:
            current_state: Current robot state
            
        Returns:
            Motor command for zero position
        """
        cmd = self.robot.create_zero_command()
        
        # Gradually transition to zero position
        ratio = min(self.current_time / self.duration_stage, 1.0)
        
        q_target = list(cmd.q_target)
        dq_target = list(cmd.dq_target)
        for i in range(g1_interface.G1_NUM_MOTOR):
            # Interpolate from current position to zero
            current_q = current_state.motor.q[i]
            target_q = 0.0
            q_target[i] = current_q * (1.0 - ratio) + target_q * ratio
            dq_target[i] = 0.0
            
        cmd.q_target = q_target
        cmd.dq_target = dq_target
        
        return cmd
        
    def create_ankle_swing_command(self) -> g1_interface.MotorCommand:
        """
        Create ankle swing command using PR mode
        
        Returns:
            Motor command for ankle swing
        """
        cmd = self.robot.create_zero_command()
        
        # Time within the ankle swing stage
        t = self.current_time - self.duration_stage
        
        # Generate sinusoidal ankle movements
        left_pitch = self.max_pitch * math.sin(2.0 * math.pi * t)
        left_roll = self.max_roll * math.sin(2.0 * math.pi * t)
        right_pitch = self.max_pitch * math.sin(2.0 * math.pi * t)
        right_roll = -self.max_roll * math.sin(2.0 * math.pi * t)  # Opposite phase

        # Set ankle targets (using PR mode indices)
        q_target = list(cmd.q_target)
        q_target[g1_interface.LeftAnklePitch] = left_pitch
        q_target[g1_interface.LeftAnkleRoll] = left_roll
        q_target[g1_interface.RightAnklePitch] = right_pitch
        q_target[g1_interface.RightAnkleRoll] = right_roll
        cmd.q_target = q_target

        # Set all dq_target to 0
        for i in range(g1_interface.G1_NUM_MOTOR):
            cmd.dq_target[i] = 0.0
            
        return cmd
        
    def print_robot_status(self, state: g1_interface.LowState, controller: g1_interface.WirelessController):
        """Print robot status information"""
        print(f"\n=== Robot Status (t={self.current_time:.1f}s, stage={self.stage}) ===")
        
        # IMU information
        rpy = state.imu.rpy
        print(f"IMU RPY: [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}] rad")
        
        # Ankle joint positions
        ankle_joints = [
            ("Left Ankle Pitch", g1_interface.LeftAnklePitch),
            ("Left Ankle Roll", g1_interface.LeftAnkleRoll),
            ("Right Ankle Pitch", g1_interface.RightAnklePitch),
            ("Right Ankle Roll", g1_interface.RightAnkleRoll),
        ]
        
        print("Ankle positions:")
        for name, idx in ankle_joints:
            pos_deg = math.degrees(state.motor.q[idx])
            vel_deg = math.degrees(state.motor.dq[idx])
            print(f"  {name}: {pos_deg:6.1f}° ({vel_deg:6.1f}°/s)")
            
        # Controller information
        print(f"Controller: L_stick=[{controller.left_stick[0]:.2f}, {controller.left_stick[1]:.2f}] "
              f"Buttons: A={controller.A} B={controller.B} X={controller.X} Y={controller.Y}")
              
        print(f"Mode machine: {state.mode_machine}")
        
    def control_loop(self):
        """Main control loop"""
        print("Starting control loop...")
        print("Stage 0: Moving to zero position (3s)")
        
        loop_count = 0
        last_print_time = 0.0
        
        while self.running:
            start_time = time.time()
            
            try:
                # Read robot state
                state = self.read_robot_state()
                controller = self.read_controller_input()
                
                # Check for emergency stop via controller
                if controller.B:  # B button for emergency stop
                    print("Emergency stop requested via B button!")
                    break
                    
                # Determine current stage
                if self.current_time < self.duration_stage:
                    # Stage 0: Move to zero position
                    if self.stage != 0:
                        self.stage = 0
                        # print(f"Stage 1: Ankle swing in PR mode (3s)")
                        
                    cmd = self.create_zero_position_command(state)
                    
                else:
                    # Stage 1: Ankle swing
                    if self.stage != 1:
                        self.stage = 1
                        # print(f"Stage 1: Ankle swing in PR mode (3s)")
                        
                    cmd = self.create_ankle_swing_command()
                    
                # Send command to robot
                self.robot.write_low_command(cmd)
                
                # Print status every second
                if self.current_time - last_print_time >= 1.0:
                    self.print_robot_status(state, controller)
                    last_print_time = self.current_time
                    
                # Update time
                self.current_time += self.control_dt
                loop_count += 1
                
                # Sleep to maintain control frequency
                elapsed_time = time.time() - start_time
                sleep_time = self.control_dt - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif loop_count % 100 == 0:  # Print warning occasionally
                    print(f"Warning: Control loop running slower than {1/self.control_dt:.0f}Hz "
                          f"(took {elapsed_time*1000:.1f}ms)")
                    
            except KeyboardInterrupt:
                print("\nKeyboard interrupt received!")
                break
            except Exception as e:
                print(f"Error in control loop: {e}")
                break
                
        print("Control loop finished")
        
    def shutdown(self):
        """Shutdown the controller"""
        print("Shutting down controller...")
        
        # Send zero command to stop the robot
        try:
            zero_cmd = self.robot.create_zero_command()
            self.robot.write_low_command(zero_cmd)
            print("Zero command sent")
        except Exception as e:
            print(f"Error sending zero command: {e}")
            
        print("Controller shutdown complete")

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python3 example_ankle_swing.py <network_interface>")
        print("Example: python3 example_ankle_swing.py eth0")
        return 1
        
    network_interface = sys.argv[1]
    
    print("=== G1 Ankle Swing Example ===")
    print(f"Network interface: {network_interface}")
    print("Control sequence:")
    print("  1. Move to zero position (3s)")
    print("  2. Ankle swing in PR mode (3s)")
    print("  3. Hold zero position")
    print("Press Ctrl+C or B button on controller to stop")
    print("")
    
    controller: Optional[G1AnkleSwingController] = None
    
    try:
        # Create controller
        controller = G1AnkleSwingController(network_interface)
        
        # Wait a moment for initialization
        time.sleep(1.0)
        
        # Run control loop
        controller.control_loop()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    finally:
        if controller is not None:
            controller.shutdown()
            
    print("Example completed")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 