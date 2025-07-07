# G1 Interface Python Binding

这是一个基于 pybind11 的 Unitree G1 机器人 Python 接口，提供了低级别的机器人控制功能。

## 功能特性

- **实时控制**: 支持 500Hz 的实时控制循环
- **数据读取**: 读取机器人状态（IMU、电机状态等）和无线控制器输入
- **命令发送**: 发送电机控制命令到机器人
- **双控制模式**: 支持 PR (Pitch/Roll) 和 AB (A/B) 控制模式
- **类型安全**: 提供完整的 Python 类型提示 (.pyi 文件)
- **线程安全**: 使用缓冲区机制确保线程安全的数据交换

## 编译

### 从主项目编译（推荐）

在主项目根目录执行：

```bash
mkdir build
cd build
cmake -DBUILD_PYTHON_BINDING=ON ..
make -j$(nproc)
```

### 单独编译

在 `python_binding` 目录中：

```bash
./build.sh
```

或手动编译：

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
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
import g1_interface

# 初始化机器人接口
robot = g1_interface.G1Interface("eth0")  # 替换为实际的网络接口名

# 读取机器人状态
state = robot.read_low_state()
print(f"IMU RPY: {state.imu.rpy}")
print(f"Joint positions: {state.motor.q}")

# 读取无线控制器状态
controller = robot.read_wireless_controller()
print(f"Left stick: {controller.left_stick}")

# 创建零位置命令
cmd = robot.create_zero_command()

# 设置目标位置（例如：左踝关节）
cmd.q_target[g1_interface.LeftAnklePitch] = 0.1  # 0.1 弧度
cmd.q_target[g1_interface.LeftAnkleRoll] = 0.0

# 发送命令到机器人
robot.write_low_command(cmd)
```

### 踝关节摆动示例

运行提供的踝关节摆动示例：

```bash
python3 example_ankle_swing.py eth0
```

这个示例演示了：
1. 机器人移动到零位置 (3秒)
2. 使用 PR 模式进行踝关节摆动 (3秒)
3. 保持零位置

## API 参考

### 主要类

#### `G1Interface`

主要的机器人控制接口类。

```python
def __init__(self, network_interface: str) -> None
def read_low_state(self) -> LowState
def read_wireless_controller(self) -> WirelessController
def write_low_command(self, command: MotorCommand) -> None
def set_control_mode(self, mode: ControlMode) -> None
def get_control_mode(self) -> ControlMode
def create_zero_command(self) -> MotorCommand
def get_default_kp(self) -> List[float]
def get_default_kd(self) -> List[float]
```

### 关节索引

机器人有29个关节，可以使用预定义的常量来访问：

```python
# 腿部关节
g1_interface.LeftHipPitch      # 0
g1_interface.LeftHipRoll       # 1
g1_interface.LeftHipYaw        # 2
g1_interface.LeftKnee          # 3
g1_interface.LeftAnklePitch    # 4
g1_interface.LeftAnkleRoll     # 5
# ... 右腿关节 6-11

# 腰部关节 12-14
g1_interface.WaistYaw          # 12
g1_interface.WaistRoll         # 13
g1_interface.WaistPitch        # 14

# 手臂关节 15-28
g1_interface.LeftShoulderPitch # 15
# ... 其他手臂关节
```

### 控制模式

```python
g1_interface.ControlMode.PR    # Pitch/Roll 模式
g1_interface.ControlMode.AB    # A/B 模式
```

## 安全注意事项

⚠️ **重要安全提示**:

- 在运行任何控制程序之前，确保机器人处于安全环境
- 始终准备紧急停止按钮
- 测试新控制算法时使用较小的运动幅度
- 监控关节温度和电压
- 使用无线控制器的 B 按钮作为紧急停止

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
   ImportError: No module named 'g1_interface'
   ```
   解决：确保编译后的 `.so` 文件在 Python 路径中

2. **网络连接失败**:
   ```
   Error: Failed to initialize DDS
   ```
   解决：检查网络接口名称是否正确，机器人是否连接

## 许可证

请参考 Unitree SDK2 的许可证条款。 