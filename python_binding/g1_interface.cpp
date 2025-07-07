#include "g1_interface.hpp"
#include <iostream>
#include <unistd.h>
#include <iomanip>  // for std::setprecision

G1Interface::G1Interface(const std::string& networkInterface)
    : mode_(PyControlMode::PR), mode_machine_(0) {
    
    // Initialize default gains
    InitDefaultGains();
    
    // Initialize DDS
    ChannelFactory::Instance()->Init(0, networkInterface);

    // Create subscribers
    lowstate_subscriber_.reset(
        new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&G1Interface::LowStateHandler, this, std::placeholders::_1), 1);

    wireless_subscriber_.reset(
        new ChannelSubscriber<unitree_go::msg::dds_::WirelessController_>(TOPIC_JOYSTICK));
    wireless_subscriber_->InitChannel(
        std::bind(&G1Interface::WirelessControllerHandler, this, std::placeholders::_1), 1);

    // Create publisher
    lowcmd_publisher_.reset(
        new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();

    // Create command writer thread
    command_writer_ptr_ = CreateRecurrentThreadEx(
        "command_writer", UT_CPU_ID_NONE, 2000, &G1Interface::LowCommandWriter, this);

    std::cout << "G1Interface initialized with network interface: " << networkInterface << std::endl;
}

G1Interface::~G1Interface() {
    if (command_writer_ptr_) {
        command_writer_ptr_.reset();
    }
    lowcmd_publisher_.reset();
    lowstate_subscriber_.reset();
    wireless_subscriber_.reset();
}

void G1Interface::InitDefaultGains() {
    // Default Kp values (same as in g1_ankle_swing_example.cpp)
    default_kp_ = {
        60, 60, 60, 100, 40, 40,      // legs
        60, 60, 60, 100, 40, 40,      // legs
        60, 40, 40,                   // waist
        40, 40, 40, 40,  40, 40, 40,  // arms
        40, 40, 40, 40,  40, 40, 40   // arms
    };

    // Default Kd values
    default_kd_ = {
        1, 1, 1, 2, 1, 1,     // legs
        1, 1, 1, 2, 1, 1,     // legs
        1, 1, 1,              // waist
        1, 1, 1, 1, 1, 1, 1,  // arms
        1, 1, 1, 1, 1, 1, 1   // arms
    };
}

void G1Interface::LowStateHandler(const void *message) {
    unitree_hg::msg::dds_::LowState_ low_state =
        *(const unitree_hg::msg::dds_::LowState_ *)message;

    if (low_state.crc() !=
        Crc32Core((uint32_t *)&low_state,
                  (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1)) {
        std::cout << "[ERROR] low_state CRC Error" << std::endl;
        return;
    }

    // Get motor state
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
        ms_tmp.q.at(i) = low_state.motor_state()[i].q();
        ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
        ms_tmp.tau_est.at(i) = low_state.motor_state()[i].tau_est();
        ms_tmp.temperature.at(i) = low_state.motor_state()[i].temperature()[0];
        ms_tmp.voltage.at(i) = low_state.motor_state()[i].vol();
    }
    motor_state_buffer_.SetData(ms_tmp);

    // Get IMU state
    ImuState imu_tmp;
    imu_tmp.omega = low_state.imu_state().gyroscope();
    imu_tmp.rpy = low_state.imu_state().rpy();
    imu_tmp.quat = low_state.imu_state().quaternion();
    imu_tmp.accel = low_state.imu_state().accelerometer();
    imu_state_buffer_.SetData(imu_tmp);

    // Update mode machine
    if (mode_machine_ != low_state.mode_machine()) {
        if (mode_machine_ == 0) {
            std::cout << "G1 type: " << unsigned(low_state.mode_machine()) << std::endl;
        }
        mode_machine_ = low_state.mode_machine();
    }
}

void G1Interface::WirelessControllerHandler(const void *message) {
    std::lock_guard<std::mutex> lock(wireless_mutex_);
    
    unitree_go::msg::dds_::WirelessController_ wireless_msg =
        *(const unitree_go::msg::dds_::WirelessController_ *)message;

    PyWirelessController controller_tmp;
    controller_tmp.left_stick[0] = wireless_msg.lx();
    controller_tmp.left_stick[1] = wireless_msg.ly();
    controller_tmp.right_stick[0] = wireless_msg.rx();
    controller_tmp.right_stick[1] = wireless_msg.ry();
    
    // Map button states (16-bit layout: bit8-11 for ABXY)
    const uint16_t keys = wireless_msg.keys();
    
    // ABXY buttons are on bits 8-11
    controller_tmp.A = (keys & 0x0100) != 0;  // bit 8
    controller_tmp.B = (keys & 0x0200) != 0;  // bit 9
    controller_tmp.X = (keys & 0x0400) != 0;  // bit 10
    controller_tmp.Y = (keys & 0x0800) != 0;  // bit 11
    
    // Other buttons
    controller_tmp.R1 = (keys & 0x0001) != 0;  // bit 0
    controller_tmp.L1 = (keys & 0x0002) != 0;  // bit 1
    controller_tmp.R2 = (keys & 0x0010) != 0;  // bit 4
    controller_tmp.L2 = (keys & 0x0020) != 0;  // bit 5

    wireless_controller_buffer_.SetData(controller_tmp);
}

void G1Interface::LowCommandWriter() {
    unitree_hg::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = static_cast<uint8_t>(mode_);
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc = motor_command_buffer_.GetData();
    if (mc) {
        for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
            dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
            dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
            dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
            dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
            dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
            dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
        }

        dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command,
                                          (sizeof(dds_low_command) >> 2) - 1);
        lowcmd_publisher_->Write(dds_low_command);
    } else {
        // Debug: Print when no command is available
        static int no_cmd_counter = 0;
        if (no_cmd_counter % 1000 == 0) {  // Print every 2 seconds
            std::cout << "[DEBUG] LowCommandWriter - No motor command available!" << std::endl;
        }
        no_cmd_counter++;
    }
}

PyLowState G1Interface::ConvertToPyLowState() {
    PyLowState py_state;
    
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();
    const std::shared_ptr<const ImuState> imu = imu_state_buffer_.GetData();
    
    if (ms) {
        py_state.motor.q = ms->q;
        py_state.motor.dq = ms->dq;
        py_state.motor.tau_est = ms->tau_est;
        py_state.motor.temperature = ms->temperature;
        py_state.motor.voltage = ms->voltage;
    }
    
    if (imu) {
        py_state.imu.rpy = imu->rpy;
        py_state.imu.omega = imu->omega;
        py_state.imu.quat = imu->quat;
        py_state.imu.accel = imu->accel;
    }
    
    py_state.mode_machine = mode_machine_;
    
    return py_state;
}

MotorCommand G1Interface::ConvertFromPyMotorCommand(const PyMotorCommand& py_cmd) {
    MotorCommand cmd;
    cmd.q_target = py_cmd.q_target;
    cmd.dq_target = py_cmd.dq_target;
    cmd.kp = py_cmd.kp;
    cmd.kd = py_cmd.kd;
    cmd.tau_ff = py_cmd.tau_ff;
    return cmd;
}

// Python interface methods
PyLowState G1Interface::ReadLowState() {
    return ConvertToPyLowState();
}

PyWirelessController G1Interface::ReadWirelessController() {
    const std::shared_ptr<const PyWirelessController> controller = 
        wireless_controller_buffer_.GetData();
    
    if (controller) {
        return *controller;
    } else {
        return PyWirelessController{};  // Return default-initialized controller
    }
}

void G1Interface::WriteLowCommand(const PyMotorCommand& command) {
    MotorCommand internal_cmd = ConvertFromPyMotorCommand(command);
    motor_command_buffer_.SetData(internal_cmd);
}

void G1Interface::SetControlMode(PyControlMode mode) {
    mode_ = mode;
}

PyControlMode G1Interface::GetControlMode() const {
    return mode_;
}

PyMotorCommand G1Interface::CreateZeroCommand() {
    PyMotorCommand cmd;
    // All arrays are already zero-initialized
    
    // Set default gains
    cmd.kp = default_kp_;
    cmd.kd = default_kd_;
    
    return cmd;
}

std::array<float, G1_NUM_MOTOR> G1Interface::GetDefaultKp() const {
    return default_kp_;
}

std::array<float, G1_NUM_MOTOR> G1Interface::GetDefaultKd() const {
    return default_kd_;
} 