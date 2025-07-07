"""
Type definitions for General Unitree Interface Python bindings
"""
from typing import List, Any, Optional
from enum import Enum

# Constants
G1_NUM_MOTOR: int
H1_NUM_MOTOR: int
H1_2_NUM_MOTOR: int

# Predefined configurations
G1_HG_CONFIG: RobotConfig
H1_GO2_CONFIG: RobotConfig
H1_2_HG_CONFIG: RobotConfig

class RobotType(Enum):
    """Robot types supported by the interface"""
    G1: int      # G1 humanoid (29 motors)
    H1: int      # H1 humanoid (19 motors)
    H1_2: int    # H1-2 humanoid (29 motors)
    CUSTOM: int  # Custom robot with specified motor count

class MessageType(Enum):
    """Message types for robot communication"""
    HG: int   # Humanoid/Go1 message format
    GO2: int  # Go2 message format

class ControlMode(Enum):
    """Control mode for robots"""
    PR: int  # Pitch/Roll mode
    AB: int  # A/B mode

class RobotConfig:
    """Robot configuration structure"""
    robot_type: RobotType
    message_type: MessageType
    num_motors: int
    name: str

    def __init__(self, robot_type: RobotType, message_type: MessageType, 
                 num_motors: int, name: str) -> None: ...

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
    q: List[float]           # Joint positions [rad] (variable length)
    dq: List[float]          # Joint velocities [rad/s] (variable length)
    tau_est: List[float]     # Estimated joint torques [N*m] (variable length)
    temperature: List[int]   # Motor temperatures [°C] (variable length)
    voltage: List[float]     # Motor voltages [V] (variable length)

    def __init__(self, num_motors: int) -> None: ...
    def __repr__(self) -> str: ...

class MotorCommand:
    """Motor command data for all joints"""
    q_target: List[float]    # Target joint positions [rad] (variable length)
    dq_target: List[float]   # Target joint velocities [rad/s] (variable length)
    kp: List[float]          # Position gains (variable length)
    kd: List[float]          # Velocity gains (variable length)
    tau_ff: List[float]      # Feedforward torques [N*m] (variable length)

    def __init__(self, num_motors: int) -> None: ...
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

    def __init__(self, num_motors: int) -> None: ...
    def __repr__(self) -> str: ...

class UnitreeInterface:
    """Main interface class for general Unitree robot control"""
    
    def __init__(self, network_interface: str, robot_type: RobotType, 
                 message_type: MessageType) -> None:
        """
        Initialize UnitreeInterface with robot type and message type
        
        Args:
            network_interface: Network interface name (e.g., "eth0", "enp2s0")
            robot_type: Type of robot (G1, H1, H1_2, CUSTOM)
            message_type: Message format (HG or GO2)
        """
        ...

    def __init__(self, network_interface: str, config: RobotConfig) -> None:
        """
        Initialize UnitreeInterface with robot configuration
        
        Args:
            network_interface: Network interface name (e.g., "eth0", "enp2s0")
            config: Robot configuration
        """
        ...

    def __init__(self, network_interface: str, robot_type: RobotType, 
                 message_type: MessageType, num_motors: int) -> None:
        """
        Initialize UnitreeInterface with custom motor count
        
        Args:
            network_interface: Network interface name (e.g., "eth0", "enp2s0")
            robot_type: Type of robot (G1, H1, H1_2, CUSTOM)
            message_type: Message format (HG or GO2)
            num_motors: Number of motors for custom robot
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
            Default position gains for all joints
        """
        ...

    def get_default_kd(self) -> List[float]:
        """
        Get default velocity gains
        
        Returns:
            Default velocity gains for all joints
        """
        ...

    def get_config(self) -> RobotConfig:
        """
        Get robot configuration
        
        Returns:
            Current robot configuration
        """
        ...

    def get_num_motors(self) -> int:
        """
        Get number of motors
        
        Returns:
            Number of motors for this robot
        """
        ...

    def get_robot_name(self) -> str:
        """
        Get robot name
        
        Returns:
            Robot name string
        """
        ...

    @staticmethod
    def create_g1(network_interface: str, message_type: MessageType = MessageType.HG) -> 'UnitreeInterface':
        """
        Create G1 robot interface
        
        Args:
            network_interface: Network interface name
            message_type: Message format (default: HG)
            
        Returns:
            UnitreeInterface instance for G1 robot
        """
        ...

    @staticmethod
    def create_h1(network_interface: str, message_type: MessageType = MessageType.GO2) -> 'UnitreeInterface':
        """
        Create H1 robot interface
        
        Args:
            network_interface: Network interface name
            message_type: Message format (default: GO2)
            
        Returns:
            UnitreeInterface instance for H1 robot
        """
        ...

    @staticmethod
    def create_h1_2(network_interface: str, message_type: MessageType = MessageType.HG) -> 'UnitreeInterface':
        """
        Create H1-2 robot interface
        
        Args:
            network_interface: Network interface name
            message_type: Message format (default: HG)
            
        Returns:
            UnitreeInterface instance for H1-2 robot
        """
        ...

    @staticmethod
    def create_custom(network_interface: str, num_motors: int, 
                     message_type: MessageType = MessageType.HG) -> 'UnitreeInterface':
        """
        Create custom robot interface
        
        Args:
            network_interface: Network interface name
            num_motors: Number of motors
            message_type: Message format (default: HG)
            
        Returns:
            UnitreeInterface instance for custom robot
        """
        ...

    def __repr__(self) -> str: ...

# Module-level functions
def create_robot(network_interface: str, robot_type: RobotType, 
                message_type: MessageType = MessageType.HG) -> UnitreeInterface:
    """
    Create robot interface based on robot type
    
    Args:
        network_interface: Network interface name
        robot_type: Type of robot
        message_type: Message format (default: HG)
        
    Returns:
        UnitreeInterface instance for specified robot
    """
    ...

def create_robot_with_config(network_interface: str, config: RobotConfig) -> UnitreeInterface:
    """
    Create robot interface with configuration
    
    Args:
        network_interface: Network interface name
        config: Robot configuration
        
    Returns:
        UnitreeInterface instance for specified configuration
    """
    ... 