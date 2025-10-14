/**
 * @file g1_arm_action_example.cpp
 * @brief This example demonstrates how to use the G1 Arm Action Client to execute predefined arm actions.
 */
#include "unitree/robot/g1/arm/g1_arm_action_error.hpp"
#include "unitree/robot/g1/arm/g1_arm_action_client.hpp"
#include <boost/program_options.hpp>

namespace po = boost::program_options;
using namespace unitree::robot;
using namespace unitree::robot::g1;

int main(int argc, const char** argv) 
{
    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1 Arm Action Example      \n\n";

    // Parse command line arguments
    po::options_description desc("Unitree G1 Arm Action Example.");
    desc.add_options()
        ("help,h", "show help message")
        ("network,n", po::value<std::string>()->default_value(""), "dds network interface")
        ("list,l", "list all supported actions")
        ("id,i", po::value<int>(), "action id to execute, 0 to list all supported actions")
        ("name", po::value<std::string>(), "custom action name to execute")
        ("stop", "stop the current custom action")
        ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if(argc < 2 || vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }

    // DDS Init
    ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    auto client = std::make_shared<G1ArmActionClient>();
    client->Init();
    // Attention: unitree actions are all less than 10s, 
    //   but the custom actions may be longer.
    client->SetTimeout(10.f);

    if(vm.count("list") || (vm.count("id") && vm["id"].as<int>() == 0)){
        std::string action_list_data;
        int32_t ret = client->GetActionList(action_list_data);
        if (ret != 0) {
            std::cerr << "Failed to get action list, error code: " << ret << "\n";
        }
        std::cout << "Available actions:\n" << action_list_data << std::endl;
    } else if (vm.count("id")) {
        int32_t ret = client->ExecuteAction(vm["id"].as<int>());
        if(ret != 0) {
            switch (ret)
            {
            case UT_ROBOT_ARM_ACTION_ERR_ARMSDK:
                std::cout << UT_ROBOT_ARM_ACTION_ERR_ARMSDK_DESC << std::endl;
                break;
            case UT_ROBOT_ARM_ACTION_ERR_HOLDING:
                std::cout << UT_ROBOT_ARM_ACTION_ERR_HOLDING_DESC << std::endl;
                break;
            case UT_ROBOT_ARM_ACTION_ERR_INVALID_ACTION_ID:
                std::cout << UT_ROBOT_ARM_ACTION_ERR_INVALID_ACTION_ID_DESC << std::endl;
                break;
            case UT_ROBOT_ARM_ACTION_ERR_INVALID_FSM_ID:
                std::cout << "The actions are only supported in fsm id {500, 501, 801}" << std::endl;
                std::cout << "You can subscribe the topic rt/sportmodestate to check the fsm id." << std::endl;
                std::cout << "And in the state 801, the actions are only supported in the fsm mode {0, 3}." << std::endl;
                std::cout << "If an error is still returned at this point, ignore this action.";
                break;
            default:
                std::cerr << "Execute action failed, error code: " << ret << std::endl;
                break;
            }
        }
    } else if (vm.count("name")) {
        int32_t ret = client->ExecuteAction(vm["name"].as<std::string>());
        if(ret != 0) std::cout << "Execute custom action failed, error code: " << ret << std::endl;
    } else if (vm.count("stop")) {
        int32_t ret = client->StopCustomAction();
    }

    return 0;
};
