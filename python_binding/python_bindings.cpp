#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "g1_interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(g1_interface, m) {
    m.doc() = "Unitree G1 Robot Interface";

    // Control mode enum
    py::enum_<PyControlMode>(m, "ControlMode")
        .value("PR", PyControlMode::PR, "Pitch/Roll mode")
        .value("AB", PyControlMode::AB, "A/B mode")
        .export_values();

    // IMU State
    py::class_<PyImuState>(m, "ImuState")
        .def(py::init<>())
        .def_readwrite("rpy", &PyImuState::rpy, "Roll, Pitch, Yaw angles [rad]")
        .def_readwrite("omega", &PyImuState::omega, "Angular velocity [rad/s]")
        .def_readwrite("quat", &PyImuState::quat, "Quaternion [w, x, y, z]")
        .def_readwrite("accel", &PyImuState::accel, "Linear acceleration [m/s^2]")
        .def("__repr__", [](const PyImuState& self) {
            return "<ImuState rpy=[" + 
                   std::to_string(self.rpy[0]) + ", " + 
                   std::to_string(self.rpy[1]) + ", " + 
                   std::to_string(self.rpy[2]) + "]>";
        });

    // Motor State
    py::class_<PyMotorState>(m, "MotorState")
        .def(py::init<>())
        .def_readwrite("q", &PyMotorState::q, "Joint positions [rad]")
        .def_readwrite("dq", &PyMotorState::dq, "Joint velocities [rad/s]")
        .def_readwrite("tau_est", &PyMotorState::tau_est, "Estimated joint torques [N*m]")
        .def_readwrite("temperature", &PyMotorState::temperature, "Motor temperatures [°C]")
        .def_readwrite("voltage", &PyMotorState::voltage, "Motor voltages [V]")
        .def("__repr__", [](const PyMotorState& self) {
            return "<MotorState with " + std::to_string(self.q.size()) + " motors>";
        });

    // Motor Command
    py::class_<PyMotorCommand>(m, "MotorCommand")
        .def(py::init<>())
        .def_readwrite("q_target", &PyMotorCommand::q_target, "Target joint positions [rad]")
        .def_readwrite("dq_target", &PyMotorCommand::dq_target, "Target joint velocities [rad/s]")
        .def_readwrite("kp", &PyMotorCommand::kp, "Position gains")
        .def_readwrite("kd", &PyMotorCommand::kd, "Velocity gains")
        .def_readwrite("tau_ff", &PyMotorCommand::tau_ff, "Feedforward torques [N*m]")
        .def("__repr__", [](const PyMotorCommand& self) {
            return "<MotorCommand with " + std::to_string(self.q_target.size()) + " motors>";
        });

    // Wireless Controller
    py::class_<PyWirelessController>(m, "WirelessController")
        .def(py::init<>())
        .def_readwrite("left_stick", &PyWirelessController::left_stick, "Left stick [x, y]")
        .def_readwrite("right_stick", &PyWirelessController::right_stick, "Right stick [x, y]")
        .def_readwrite("A", &PyWirelessController::A, "A button")
        .def_readwrite("B", &PyWirelessController::B, "B button")
        .def_readwrite("X", &PyWirelessController::X, "X button")
        .def_readwrite("Y", &PyWirelessController::Y, "Y button")
        .def_readwrite("L1", &PyWirelessController::L1, "L1 button")
        .def_readwrite("L2", &PyWirelessController::L2, "L2 button")
        .def_readwrite("R1", &PyWirelessController::R1, "R1 button")
        .def_readwrite("R2", &PyWirelessController::R2, "R2 button")
        .def("__repr__", [](const PyWirelessController& self) {
            return "<WirelessController left_stick=[" + 
                   std::to_string(self.left_stick[0]) + ", " + 
                   std::to_string(self.left_stick[1]) + "] " +
                   "right_stick=[" +
                   std::to_string(self.right_stick[0]) + ", " +
                   std::to_string(self.right_stick[1]) + "] " +
                   "A=" + std::to_string(self.A) + " " +
                   "B=" + std::to_string(self.B) + " " +
                   "X=" + std::to_string(self.X) + " " +
                   "Y=" + std::to_string(self.Y) + " " +
                   "L1=" + std::to_string(self.L1) + " " +
                   "L2=" + std::to_string(self.L2) + " " +
                   "R1=" + std::to_string(self.R1) + " " +
                   "R2=" + std::to_string(self.R2) + ">";
        });

    // Low State
    py::class_<PyLowState>(m, "LowState")
        .def(py::init<>())
        .def_readwrite("imu", &PyLowState::imu, "IMU state")
        .def_readwrite("motor", &PyLowState::motor, "Motor state")
        .def_readwrite("mode_machine", &PyLowState::mode_machine, "Robot mode machine state")
        .def("__repr__", [](const PyLowState& self) {
            return "<LowState mode_machine=" + std::to_string(self.mode_machine) + ">";
        });

    // Main G1Interface class
    py::class_<G1Interface>(m, "G1Interface")
        .def(py::init<const std::string&>(), 
             py::arg("network_interface"), 
             "Initialize G1Interface with network interface name")
        .def("read_low_state", &G1Interface::ReadLowState,
             "Read current robot low state")
        .def("read_wireless_controller", &G1Interface::ReadWirelessController,
             "Read current wireless controller state")
        .def("write_low_command", &G1Interface::WriteLowCommand,
             py::arg("command"),
             "Write motor command to robot")
        .def("set_control_mode", &G1Interface::SetControlMode,
             py::arg("mode"),
             "Set control mode (PR or AB)")
        .def("get_control_mode", &G1Interface::GetControlMode,
             "Get current control mode")
        .def("create_zero_command", &G1Interface::CreateZeroCommand,
             "Create a zero motor command with default gains")
        .def("get_default_kp", &G1Interface::GetDefaultKp,
             "Get default position gains")
        .def("get_default_kd", &G1Interface::GetDefaultKd,
             "Get default velocity gains")
        .def("__repr__", [](const G1Interface& self) {
            return "<G1Interface>";
        });

    // Add constants
    m.attr("G1_NUM_MOTOR") = G1_NUM_MOTOR;
    
    // Joint indices as constants
    m.attr("LeftHipPitch") = 0;
    m.attr("LeftHipRoll") = 1;
    m.attr("LeftHipYaw") = 2;
    m.attr("LeftKnee") = 3;
    m.attr("LeftAnklePitch") = 4;
    m.attr("LeftAnkleB") = 4;
    m.attr("LeftAnkleRoll") = 5;
    m.attr("LeftAnkleA") = 5;
    m.attr("RightHipPitch") = 6;
    m.attr("RightHipRoll") = 7;
    m.attr("RightHipYaw") = 8;
    m.attr("RightKnee") = 9;
    m.attr("RightAnklePitch") = 10;
    m.attr("RightAnkleB") = 10;
    m.attr("RightAnkleRoll") = 11;
    m.attr("RightAnkleA") = 11;
    m.attr("WaistYaw") = 12;
    m.attr("WaistRoll") = 13;
    m.attr("WaistA") = 13;
    m.attr("WaistPitch") = 14;
    m.attr("WaistB") = 14;
    m.attr("LeftShoulderPitch") = 15;
    m.attr("LeftShoulderRoll") = 16;
    m.attr("LeftShoulderYaw") = 17;
    m.attr("LeftElbow") = 18;
    m.attr("LeftWristRoll") = 19;
    m.attr("LeftWristPitch") = 20;
    m.attr("LeftWristYaw") = 21;
    m.attr("RightShoulderPitch") = 22;
    m.attr("RightShoulderRoll") = 23;
    m.attr("RightShoulderYaw") = 24;
    m.attr("RightElbow") = 25;
    m.attr("RightWristRoll") = 26;
    m.attr("RightWristPitch") = 27;
    m.attr("RightWristYaw") = 28;
} 