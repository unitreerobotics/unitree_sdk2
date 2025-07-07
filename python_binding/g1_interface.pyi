"""
Type definitions for G1 Interface Python bindings
"""
from typing import List, Any
from enum import Enum

# Constants
G1_NUM_MOTOR: int

# Joint indices
LeftHipPitch: int
LeftHipRoll: int
LeftHipYaw: int
LeftKnee: int
LeftAnklePitch: int
LeftAnkleB: int
LeftAnkleRoll: int
LeftAnkleA: int
RightHipPitch: int
RightHipRoll: int
RightHipYaw: int
RightKnee: int
RightAnklePitch: int
RightAnkleB: int
RightAnkleRoll: int
RightAnkleA: int
WaistYaw: int
WaistRoll: int
WaistA: int
WaistPitch: int
WaistB: int
LeftShoulderPitch: int
LeftShoulderRoll: int
LeftShoulderYaw: int
LeftElbow: int
LeftWristRoll: int
LeftWristPitch: int
LeftWristYaw: int
RightShoulderPitch: int
RightShoulderRoll: int
RightShoulderYaw: int
RightElbow: int
RightWristRoll: int
RightWristPitch: int
RightWristYaw: int

class ControlMode(Enum):
    """Control mode for G1 robot"""
    PR: int  # Pitch/Roll mode
    AB: int  # A/B mode

class ImuState:
    """IMU state data"""
    rpy: List[float]      # Roll, Pitch, Yaw angles [rad] (3 elements)
    omega: List[float]    # Angular velocity [rad/s] (3 elements)
    quat: List[float]     # Quaternion [w, x, y, z] (4 elements) 
    accel: List[float]    # Linear acceleration [m/s^2] (3 elements)

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class MotorState:
    """Motor state data for all joints"""
    q: List[float]           # Joint positions [rad] (29 elements)
    dq: List[float]          # Joint velocities [rad/s] (29 elements)
    tau_est: List[float]     # Estimated joint torques [N*m] (29 elements)
    temperature: List[int]   # Motor temperatures [°C] (29 elements)
    voltage: List[float]     # Motor voltages [V] (29 elements)

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class MotorCommand:
    """Motor command data for all joints"""
    q_target: List[float]    # Target joint positions [rad] (29 elements)
    dq_target: List[float]   # Target joint velocities [rad/s] (29 elements)
    kp: List[float]          # Position gains (29 elements)
    kd: List[float]          # Velocity gains (29 elements)
    tau_ff: List[float]      # Feedforward torques [N*m] (29 elements)

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class WirelessController:
    """Wireless controller state"""
    left_stick: List[float]  # Left stick [x, y] (2 elements)
    right_stick: List[float] # Right stick [x, y] (2 elements)
    A: bool                  # A button
    B: bool                  # B button
    X: bool                  # X button
    Y: bool                  # Y button
    L1: bool                 # L1 button
    L2: bool                 # L2 button
    R1: bool                 # R1 button
    R2: bool                 # R2 button

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class LowState:
    """Complete robot low-level state"""
    imu: ImuState            # IMU state
    motor: MotorState        # Motor state
    mode_machine: int        # Robot mode machine state

    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...

class G1Interface:
    """Main interface class for G1 robot control"""
    
    def __init__(self, network_interface: str) -> None:
        """
        Initialize G1Interface with network interface name
        
        Args:
            network_interface: Network interface name (e.g., "eth0", "enp2s0")
        """
        ...

    def read_low_state(self) -> LowState:
        """
        Read current robot low state
        
        Returns:
            Current robot state including IMU and motor data
        """
        ...

    def read_wireless_controller(self) -> WirelessController:
        """
        Read current wireless controller state
        
        Returns:
            Current controller button and stick states
        """
        ...

    def write_low_command(self, command: MotorCommand) -> None:
        """
        Write motor command to robot
        
        Args:
            command: Motor command to send to robot
        """
        ...

    def set_control_mode(self, mode: ControlMode) -> None:
        """
        Set control mode (PR or AB)
        
        Args:
            mode: Control mode to set
        """
        ...

    def get_control_mode(self) -> ControlMode:
        """
        Get current control mode
        
        Returns:
            Current control mode
        """
        ...

    def create_zero_command(self) -> MotorCommand:
        """
        Create a zero motor command with default gains
        
        Returns:
            Zero command with appropriate default PD gains
        """
        ...

    def get_default_kp(self) -> List[float]:
        """
        Get default position gains
        
        Returns:
            Default position gains for all joints (29 elements)
        """
        ...

    def get_default_kd(self) -> List[float]:
        """
        Get default velocity gains
        
        Returns:
            Default velocity gains for all joints (29 elements)
        """
        ...

    def __repr__(self) -> str: ... 