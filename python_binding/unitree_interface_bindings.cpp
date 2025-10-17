#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "unitree_interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(unitree_interface, m) {
    m.doc() = "General Unitree robot interface supporting both HG and GO2 message types";
    
    // Enums
    py::enum_<RobotType>(m, "RobotType")
        .value("G1", RobotType::G1)
        .value("H1", RobotType::H1)
        .value("H1_2", RobotType::H1_2)
        .value("GO2", RobotType::GO2)
        .value("CUSTOM", RobotType::CUSTOM)
        .export_values();
    
    py::enum_<MessageType>(m, "MessageType")
        .value("HG", MessageType::HG)
        .value("GO2", MessageType::GO2)
        .export_values();
    
    py::enum_<PyControlMode>(m, "ControlMode")
        .value("PR", PyControlMode::PR)
        .value("AB", PyControlMode::AB)
        .export_values();
    
    // Data structures
    py::class_<PyImuState>(m, "ImuState")
        .def(py::init<>())
        .def_readwrite("rpy", &PyImuState::rpy)
        .def_readwrite("omega", &PyImuState::omega)
        .def_readwrite("quat", &PyImuState::quat)
        .def_readwrite("accel", &PyImuState::accel);
    
    py::class_<PyMotorState>(m, "MotorState")
        .def(py::init<int>())
        .def_readwrite("q", &PyMotorState::q)
        .def_readwrite("dq", &PyMotorState::dq)
        .def_readwrite("tau_est", &PyMotorState::tau_est)
        .def_readwrite("temperature", &PyMotorState::temperature)
        .def_readwrite("voltage", &PyMotorState::voltage);
    
    py::class_<PyMotorCommand>(m, "MotorCommand")
        .def(py::init<int>())
        .def_readwrite("q_target", &PyMotorCommand::q_target)
        .def_readwrite("dq_target", &PyMotorCommand::dq_target)
        .def_readwrite("kp", &PyMotorCommand::kp)
        .def_readwrite("kd", &PyMotorCommand::kd)
        .def_readwrite("tau_ff", &PyMotorCommand::tau_ff);
    
    py::class_<PyWirelessController>(m, "WirelessController")
        .def(py::init<>())
        .def_readwrite("lx", &PyWirelessController::lx)
        .def_readwrite("ly", &PyWirelessController::ly)
        .def_readwrite("rx", &PyWirelessController::rx)
        .def_readwrite("ry", &PyWirelessController::ry)
        .def_readwrite("keys", &PyWirelessController::keys);
    
    py::class_<PyLowState>(m, "LowState")
        .def(py::init<int>())
        .def_readwrite("imu", &PyLowState::imu)
        .def_readwrite("motor", &PyLowState::motor)
        .def_readwrite("mode_machine", &PyLowState::mode_machine);
    
    py::class_<RobotConfig>(m, "RobotConfig")
        .def(py::init<RobotType, MessageType, int, const std::string&>())
        .def_readwrite("robot_type", &RobotConfig::robot_type)
        .def_readwrite("message_type", &RobotConfig::message_type)
        .def_readwrite("num_motors", &RobotConfig::num_motors)
        .def_readwrite("name", &RobotConfig::name);
    
    // Main interface class
    py::class_<UnitreeInterface, std::shared_ptr<UnitreeInterface>>(m, "UnitreeInterface")
        // Constructors
        .def(py::init<const std::string&, RobotType, MessageType>())
        .def(py::init<const std::string&, const RobotConfig&>())
        .def(py::init<const std::string&, RobotType, MessageType, int>())
        
        // Python interface methods
        .def("read_low_state", &UnitreeInterface::ReadLowState)
        .def("read_wireless_controller", &UnitreeInterface::ReadWirelessController)
        .def("write_low_command", static_cast<void(UnitreeInterface::*)(const PyMotorCommand&)>(&UnitreeInterface::WriteLowCommand))
        .def("set_control_mode", &UnitreeInterface::SetControlMode)
        .def("get_control_mode", &UnitreeInterface::GetControlMode)
        
        // Utility methods
        .def("create_zero_command", &UnitreeInterface::CreateZeroCommand)
        .def("get_default_kp", &UnitreeInterface::GetDefaultKp)
        .def("get_default_kd", &UnitreeInterface::GetDefaultKd)
        
        // Configuration methods
        .def("get_config", &UnitreeInterface::GetConfig)
        .def("get_num_motors", &UnitreeInterface::GetNumMotors)
        .def("get_robot_name", static_cast<std::string(UnitreeInterface::*)() const>(&UnitreeInterface::GetRobotName))
        
        // Static factory methods
        .def_static("create_g1", &UnitreeInterface::CreateG1, 
                   py::arg("network_interface"), py::arg("message_type") = MessageType::HG)
        .def_static("create_h1", &UnitreeInterface::CreateH1,
                   py::arg("network_interface"), py::arg("message_type") = MessageType::GO2)
        .def_static("create_h1_2", &UnitreeInterface::CreateH1_2,
                   py::arg("network_interface"), py::arg("message_type") = MessageType::HG)
        .def_static("create_go2", &UnitreeInterface::CreateGO2,
                   py::arg("network_interface"), py::arg("message_type") = MessageType::GO2)
        .def_static("create_custom", &UnitreeInterface::CreateCustom,
                   py::arg("network_interface"), py::arg("num_motors"), 
                   py::arg("message_type") = MessageType::HG);
    
    // Predefined configurations (expose as module attributes since RobotConfigs is a namespace)
    m.attr("G1_HG_CONFIG") = RobotConfigs::G1_HG;
    m.attr("H1_GO2_CONFIG") = RobotConfigs::H1_GO2;
    m.attr("H1_2_HG_CONFIG") = RobotConfigs::H1_2_HG;
    m.attr("GO2_GO2_CONFIG") = RobotConfigs::GO2_GO2;
    
    // Module-level functions for convenience
    m.def("create_robot", [](const std::string& network_interface, RobotType robot_type, 
                            MessageType message_type = MessageType::HG) {
        switch (robot_type) {
            case RobotType::G1: return UnitreeInterface::CreateG1(network_interface, message_type);
            case RobotType::H1: return UnitreeInterface::CreateH1(network_interface, message_type);
            case RobotType::H1_2: return UnitreeInterface::CreateH1_2(network_interface, message_type);
            case RobotType::GO2: return UnitreeInterface::CreateGO2(network_interface, message_type);
            default: throw std::runtime_error("Unknown robot type");
        }
    }, py::arg("network_interface"), py::arg("robot_type"), py::arg("message_type") = MessageType::HG);
    
    m.def("create_robot_with_config", [](const std::string& network_interface, const RobotConfig& config) {
        return std::make_shared<UnitreeInterface>(network_interface, config);
    }, py::arg("network_interface"), py::arg("config"));
    
    // Constants
    m.attr("G1_NUM_MOTOR") = 29;
    m.attr("H1_NUM_MOTOR") = 19;
    m.attr("H1_2_NUM_MOTOR") = 29;
    m.attr("GO2_NUM_MOTOR") = 12;
}
