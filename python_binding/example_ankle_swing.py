#!/usr/bin/env python3
"""
General Unitree Interface Example

This example demonstrates how to use the general Unitree interface
with different robot types (G1, H1, H1_2) and message formats (HG, GO2).
"""

import sys
import time
import math
import signal
from typing import Optional

sys.path.append("/home/ANT.AMAZON.COM/zyuanhan/Humanoid/falcon_deploy/unitree_sdk2/build/lib")

try:
    import unitree_interface
except ImportError:
    print("Error: unitree_interface module not found!")
    print("Please build the module first using: ./build.sh")
    print("Or make sure the compiled .so file is in your Python path")
    sys.exit(1)

class GeneralUnitreeController:
    """General Unitree robot controller supporting multiple robot types"""
    
    def __init__(self, network_interface: str, robot_type: unitree_interface.RobotType, 
                 message_type: unitree_interface.MessageType = unitree_interface.MessageType.HG):
        """
        Initialize the controller
        
        Args:
            network_interface: Network interface name (e.g., "eth0", "lo")
            robot_type: Type of robot (G1, H1, H1_2)
            message_type: Message format (HG or GO2)
        """
        print(f"Initializing {robot_type} robot with {message_type} messages on interface: {network_interface}")
        
        # Create robot interface
        self.robot = unitree_interface.create_robot(network_interface, robot_type, message_type)
        self.running = True
        
        # Control parameters
        self.control_dt = 0.002  # 2ms control loop, 500Hz
        self.duration_stage = 3.0  # 3 seconds per stage
        self.current_time = 0.0
        self.stage = 0  # 0: init to zero, 1: joint swing, 2: finished
        
        # Get robot configuration
        self.config = self.robot.get_config()
        self.num_motors = self.robot.get_num_motors()
        
        print(f"Robot: {self.config.name}")
        print(f"Motors: {self.num_motors}")
        print(f"Message type: {self.config.message_type}")
        
        # Set control mode to PR (Pitch/Roll)
        self.robot.set_control_mode(unitree_interface.ControlMode.PR)
        control_mode = self.robot.get_control_mode()
        print(f"Control mode set to: {'PR' if control_mode == unitree_interface.ControlMode.PR else 'AB'}")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        self.running = False
        
    def read_robot_state(self) -> unitree_interface.LowState:
        """Read current robot state"""
        return self.robot.read_low_state()
        
    def read_controller_input(self) -> unitree_interface.WirelessController:
        """Read wireless controller input"""
        return self.robot.read_wireless_controller()
        
    def create_zero_position_command(self, current_state: unitree_interface.LowState) -> unitree_interface.MotorCommand:
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
        
        for i in range(self.num_motors):
            # Interpolate from current position to zero
            current_q = current_state.motor.q[i]
            target_q = 0.0
            cmd.q_target[i] = current_q * (1.0 - ratio) + target_q * ratio
            cmd.dq_target[i] = 0.0
            
        return cmd
        
    def create_joint_swing_command(self) -> unitree_interface.MotorCommand:
        """
        Create joint swing command for demonstration
        
        Returns:
            Motor command for joint swing
        """
        cmd = self.robot.create_zero_command()
        # Time within the joint swing stage
        t = self.current_time - self.duration_stage

        # Generate sinusoidal joint movements for different robot types
        if self.config.robot_type == unitree_interface.RobotType.G1:
            # G1: Ankle swing for humanoid (29 motors)
            q_target = list(cmd.q_target)
            max_amplitude = math.radians(20.0)
            for i in range(4, 6):  # Left ankle joints
                q_target[i] = max_amplitude * math.sin(2.0 * math.pi * t)
            for i in range(10, 12):  # Right ankle joints
                q_target[i] = max_amplitude * math.sin(2.0 * math.pi * t)
            cmd.q_target = q_target
                
        elif self.config.robot_type == unitree_interface.RobotType.H1:
            # H1: General joint swing for humanoid (19 motors)
            q_target = list(cmd.q_target)
            max_amplitude = math.radians(15.0)
            for i in range(min(6, self.num_motors)):
                q_target[i] = max_amplitude * math.sin(2.0 * math.pi * t + i * 0.5)
            cmd.q_target = q_target
                
        elif self.config.robot_type == unitree_interface.RobotType.H1_2:
            # H1-2: Ankle swing for humanoid (29 motors)
            q_target = list(cmd.q_target)
            max_amplitude = math.radians(20.0)
            for i in range(4, 6):  # Left ankle joints
                q_target[i] = max_amplitude * math.sin(2.0 * math.pi * t)
            for i in range(10, 12):  # Right ankle joints
                q_target[i] = max_amplitude * math.sin(2.0 * math.pi * t)
            cmd.q_target = q_target

        # Set all dq_target to 0 (position control)
        dq_target = list(cmd.dq_target)
        for i in range(self.num_motors):
            dq_target[i] = 0.0
        cmd.dq_target = dq_target
            
        return cmd
        
    def print_robot_status(self, state: unitree_interface.LowState, controller: unitree_interface.WirelessController):
        """Print robot status information"""
        print(f"\n=== {self.config.name} Status (t={self.current_time:.1f}s, stage={self.stage}) ===")
        
        # IMU information
        rpy = state.imu.rpy
        print(f"IMU RPY: [{rpy[0]:.3f}, {rpy[1]:.3f}, {rpy[2]:.3f}] rad")
        
        # Motor positions (show first few motors)
        num_to_show = min(6, self.num_motors)
        print(f"Motor positions (first {num_to_show}):")
        for i in range(num_to_show):
            pos_deg = math.degrees(state.motor.q[i])
            vel_deg = math.degrees(state.motor.dq[i])
            print(f"  Motor {i}: {pos_deg:6.1f}° ({vel_deg:6.1f}°/s)")
            
        # Controller information
        print(f"Controller: L_stick=[{controller.lx:.2f}, {controller.ly:.2f}] "
              f"Buttons: A={controller.keys == 0x0100} B={controller.keys == 0x0200} "
              f"X={controller.keys == 0x0400} Y={controller.keys == 0x0800}")
              
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
                if controller.keys == 0x0200:  # B button for emergency stop
                    print("Emergency stop requested via B button!")
                    break
                    
                # Check for controller-based movement (A button)
                # Normal control sequence
                # Determine current stage
                if self.current_time < self.duration_stage:
                    # Stage 0: Move to zero position
                    if self.stage != 0:
                        self.stage = 0
                        
                    cmd = self.create_zero_position_command(state)
                    
                else:
                    # Stage 1: Joint swing
                    if self.stage != 1:
                        self.stage = 1
                        print(f"Stage 1: Joint swing demonstration")
                        print(f"  Robot type: {self.config.robot_type}")
                        print(f"  Number of motors: {self.num_motors}")
                        print(f"  Message type: {self.config.message_type}")
                        
                    cmd = self.create_joint_swing_command()
                    
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
    if len(sys.argv) < 3:
        print("Usage: python3 example_general_interface.py <network_interface> <robot_type> [message_type]")
        print("Robot types: G1, H1, H1_2")
        print("Message types: HG, GO2")
        print("Note: H1 uses GO2 messages by default, G1 and H1_2 use HG messages")
        print("Examples:")
        print("  python3 example_general_interface.py eth0 G1")
        print("  python3 example_general_interface.py eth0 H1 GO2")
        print("  python3 example_general_interface.py lo H1_2")
        return 1
        
    network_interface = sys.argv[1]
    robot_type_str = sys.argv[2].upper()
    message_type_str = sys.argv[3].upper() if len(sys.argv) > 3 else "HG"
    
    # Parse robot type
    robot_type_map = {
        "G1": unitree_interface.RobotType.G1,
        "H1": unitree_interface.RobotType.H1,
        "H1_2": unitree_interface.RobotType.H1_2
    }
    
    if robot_type_str not in robot_type_map:
        print(f"Error: Unknown robot type '{robot_type_str}'")
        print("Supported types: G1, H1, H1_2")
        return 1
    
    robot_type = robot_type_map[robot_type_str]
    
    # Parse message type
    message_type_map = {
        "HG": unitree_interface.MessageType.HG,
        "GO2": unitree_interface.MessageType.GO2
    }
    
    if message_type_str not in message_type_map:
        print(f"Error: Unknown message type '{message_type_str}'")
        print("Supported types: HG, GO2")
        return 1
    
    message_type = message_type_map[message_type_str]
    
    print("=== General Unitree Interface Example ===")
    print(f"Network interface: {network_interface}")
    print(f"Robot type: {robot_type_str}")
    print(f"Message type: {message_type_str}")
    print("Control sequence:")
    print("  1. Move to zero position (3s)")
    print("  2. Joint swing demonstration (continuous)")
    print("Controller options:")
    print("  - A button: Manual control with joysticks")
    print("  - B button: Emergency stop")
    print("Press Ctrl+C to stop")
    print("")
    
    controller: Optional[GeneralUnitreeController] = None
    
    try:
        # Create controller
        controller = GeneralUnitreeController(network_interface, robot_type, message_type)
        
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