# Unitree Interface Python Binding

这是一个基于 pybind11 的通用 Unitree 机器人 Python 接口，支持多种机器人类型（G1、H1、H1-2）和消息格式（HG、GO2）。

## 功能特性

- **多机器人支持**: 支持 G1、H1、H1-2 等多种机器人类型
- **多消息格式**: 支持 HG 和 GO2 消息格式
- **实时控制**: 支持 500Hz 的实时控制循环
- **数据读取**: 读取机器人状态（IMU、电机状态等）和无线控制器输入
- **命令发送**: 发送电机控制命令到机器人
- **双控制模式**: 支持 PR (Pitch/Roll) 和 AB (A/B) 控制模式
- **类型安全**: 提供完整的 Python 类型提示 (.pyi 文件)
- **线程安全**: 使用缓冲区机制确保线程安全的数据交换
- **工厂方法**: 提供便捷的机器人创建方法

## 支持的机器人类型

| 机器人类型 | 电机数量 | 默认消息格式 | 描述 |
|-----------|---------|-------------|------|
| G1        | 29      | HG          | G1 人形机器人 |
| H1        | 19      | GO2         | H1 人形机器人 |
| H1-2      | 29      | HG          | H1-2 人形机器人 |
| CUSTOM    | 自定义   | HG          | 自定义机器人配置 |

## 编译

### 从主项目编译（推荐）

在主项目根目录执行：

```bash
mkdir build
cd build
cmake -DBUILD_PYTHON_BINDING=ON -DCMAKE_BUILD_TYPE=Release .. 
make -j$(nproc)
```

### 单独编译

在 `python_binding` 目录中：

```bash
./build.sh --sdk-path /opt/unitree_sdk2
```

## 系统要求

- Ubuntu 18.04/20.04/22.04 或兼容系统
- Python 3.6+
- CMake 3.12+
- GCC 7+ 或 Clang 6+
- Unitree SDK2
- pybind11

## 安装依赖

```bash
# 安装基本依赖
sudo apt-get update
sudo apt-get install build-essential cmake python3-dev python3-pip

# 安装 Python 依赖
pip3 install pybind11 pybind11-stubgen numpy
```

## 使用方法

### 基本使用

```python
import unitree_interface

# 方法1: 使用工厂方法创建机器人
robot = unitree_interface.create_robot("eth0", unitree_interface.RobotType.G1)

# 方法2: 直接创建接口
robot = unitree_interface.UnitreeInterface("eth0", unitree_interface.RobotType.H1, unitree_interface.MessageType.GO2)

# 方法3: 使用预定义配置
robot = unitree_interface.UnitreeInterface("eth0", unitree_interface.RobotConfigs.G1_HG)

# 读取机器人状态
state = robot.read_low_state()
print(f"IMU RPY: {state.imu.rpy}")
print(f"Joint positions: {state.motor.q}")

# 读取无线控制器状态
controller = robot.read_wireless_controller()
print(f"Left stick: {controller.left_stick}")

# 创建零位置命令
cmd = robot.create_zero_command()

# 设置目标位置（根据机器人类型调整关节索引）
num_motors = robot.get_num_motors()
if num_motors > 4:
    cmd.q_target[4] = 0.1  # 左踝关节俯仰
if num_motors > 5:
    cmd.q_target[5] = 0.0  # 左踝关节横滚

# 发送命令到机器人
robot.write_low_command(cmd)
```

### 通用接口示例

运行提供的通用接口示例：

```bash
# G1 机器人示例
python3 example_general_interface.py eth0 G1

# H1 机器人示例
python3 example_general_interface.py eth0 H1 GO2

# H1-2 机器人示例
python3 example_general_interface.py eth0 H1_2
```

这个示例演示了：
1. 机器人移动到零位置 (3秒)
2. 关节摆动演示（连续）
3. 控制器手动控制（按 A 按钮）

## API 参考

### 主要类

#### `UnitreeInterface`

主要的机器人控制接口类。

```python
def __init__(self, network_interface: str, robot_type: RobotType, message_type: MessageType = MessageType.HG) -> None
def __init__(self, network_interface: str, config: RobotConfig) -> None
def __init__(self, network_interface: str, robot_type: RobotType, message_type: MessageType, num_motors: int) -> None

def read_low_state(self) -> LowState
def read_wireless_controller(self) -> WirelessController
def write_low_command(self, command: MotorCommand) -> None
def set_control_mode(self, mode: ControlMode) -> None
def get_control_mode(self) -> ControlMode
def create_zero_command(self) -> MotorCommand
def get_default_kp(self) -> List[float]
def get_default_kd(self) -> List[float]
def get_config(self) -> RobotConfig
def get_num_motors(self) -> int
def get_robot_name(self) -> str
```

### 工厂方法

```python
# 便捷的机器人创建方法
unitree_interface.create_g1(network_interface: str, message_type: MessageType = MessageType.HG) -> UnitreeInterface
unitree_interface.create_h1(network_interface: str, message_type: MessageType = MessageType.GO2) -> UnitreeInterface
unitree_interface.create_h1_2(network_interface: str, message_type: MessageType = MessageType.HG) -> UnitreeInterface
unitree_interface.create_custom(network_interface: str, num_motors: int, message_type: MessageType = MessageType.HG) -> UnitreeInterface

# 通用创建方法
unitree_interface.create_robot(network_interface: str, robot_type: RobotType, message_type: MessageType = MessageType.HG) -> UnitreeInterface
```

### 枚举类型

```python
class RobotType(Enum):
    G1 = 0      # G1 人形机器人 (29 电机)
    H1 = 1      # H1 人形机器人 (19 电机)
    H1_2 = 2    # H1-2 人形机器人 (29 电机)
    CUSTOM = 99 # 自定义机器人

class MessageType(Enum):
    HG = 0   # Humanoid/Go1 消息格式
    GO2 = 1  # Go2 消息格式

class ControlMode(Enum):
    PR = 0  # Pitch/Roll 模式
    AB = 1  # A/B 模式
```

### 预定义配置

```python
unitree_interface.RobotConfigs.G1_HG    # G1 + HG 消息
unitree_interface.RobotConfigs.H1_GO2   # H1 + GO2 消息
unitree_interface.RobotConfigs.H1_2_HG  # H1-2 + HG 消息
```

### 数据结构

```python
class LowState:
    imu: ImuState           # IMU 状态
    motor: MotorState       # 电机状态
    mode_machine: int       # 机器人模式机状态

class MotorState:
    q: List[float]          # 关节位置 [rad]
    dq: List[float]         # 关节速度 [rad/s]
    tau_est: List[float]    # 估计关节力矩 [N*m]
    temperature: List[int]  # 电机温度 [°C]
    voltage: List[float]    # 电机电压 [V]

class MotorCommand:
    q_target: List[float]   # 目标关节位置 [rad]
    dq_target: List[float]  # 目标关节速度 [rad/s]
    kp: List[float]         # 位置增益
    kd: List[float]         # 速度增益
    tau_ff: List[float]     # 前馈力矩 [N*m]

class WirelessController:
    left_stick: List[float]  # 左摇杆 [x, y]
    right_stick: List[float] # 右摇杆 [x, y]
    A: bool                  # A 按钮
    B: bool                  # B 按钮
    X: bool                  # X 按钮
    Y: bool                  # Y 按钮
    L1: bool                 # L1 按钮
    L2: bool                 # L2 按钮
    R1: bool                 # R1 按钮
    R2: bool                 # R2 按钮
```

## 安全注意事项

⚠️ **重要安全提示**:

- 在运行任何控制程序之前，确保机器人处于安全环境
- 始终准备紧急停止按钮
- 测试新控制算法时使用较小的运动幅度
- 监控关节温度和电压
- 使用无线控制器的 B 按钮作为紧急停止
- 确保使用正确的消息格式（HG vs GO2）

## 故障排除

### 编译错误

1. **找不到 Unitree SDK**:
   ```
   CMake Error: UNITREE_SDK not found
   ```
   解决：确保在主项目根目录编译，或检查SDK路径

2. **找不到 pybind11**:
   ```
   CMake Error: pybind11 not found
   ```
   解决：`pip install pybind11`

### 运行时错误

1. **模块导入失败**:
   ```python
   ImportError: No module named 'unitree_interface'
   ```
   解决：确保编译后的 `.so` 文件在 Python 路径中

2. **网络连接失败**:
   ```
   Error: Failed to initialize DDS
   ```
   解决：检查网络接口名称是否正确，机器人是否连接

3. **消息格式不匹配**:
   ```
   Error: Message type mismatch
   ```
   解决：确保使用正确的消息格式（H1 使用 GO2，G1/H1-2 使用 HG）

### 机器人类型选择

- **G1**: 29 个电机，使用 HG 消息格式
- **H1**: 19 个电机，使用 GO2 消息格式
- **H1-2**: 29 个电机，使用 HG 消息格式

## 许可证

请参考 Unitree SDK2 的许可证条款。 