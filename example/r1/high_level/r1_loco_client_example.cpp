#include <chrono>
#include <iostream>
#include <thread>

#include "unitree/robot/r1/loco/r1_loco_api.hpp"
#include "unitree/robot/r1/loco/r1_loco_client.hpp"

std::vector<float> stringToFloatVector(const std::string &str) {
  std::vector<float> result;
  std::stringstream ss(str);
  float num;
  while (ss >> num) {
    result.push_back(num);
    // ignore any trailing whitespace
    ss.ignore();
  }
  return result;
}

int main(int argc, char const *argv[]) {
  std::map<std::string, std::string> args = {{"network_interface", "lo"}};

  std::map<std::string, std::string> values;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      size_t pos = arg.find("=");
      std::string key, value;
      if (pos != std::string::npos) {
        key = arg.substr(2, pos - 2);
        value = arg.substr(pos + 1);

        if (value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.length() - 2);
        }
      } else {
        key = arg.substr(2);
        value = "";
      }
      if (args.find(key) != args.end()) {
        args[key] = value;
      } else {
        args.insert({{key, value}});
      }
    }
  }
  std::string network_interface = args["network_interface"];
  unitree::robot::ChannelFactory::Instance()->Init(0, network_interface);

  unitree::robot::r1::LocoClient client;

  client.Init();
  client.SetTimeout(10.f);

  for (const auto &arg_pair : args) {
    std::cout << "Processing command: [" << arg_pair.first << "] with param: [" << arg_pair.second << "] ..."
              << std::endl;
    if (arg_pair.first == "network_interface") {
      continue;
    }

    if (arg_pair.first == "get_fsm_id") {
      int fsm_id;
      client.GetFsmId(fsm_id);
      std::cout << "current fsm_id: " << fsm_id << std::endl;
    }

    if (arg_pair.first == "get_fsm_mode") {
      int fsm_mode;
      client.GetFsmMode(fsm_mode);
      std::cout << "current fsm_mode: " << fsm_mode << std::endl;
    }

    if (arg_pair.first == "set_fsm_id") {
      int fsm_id = std::stoi(arg_pair.second);
      client.SetFsmId(fsm_id);
      std::cout << "set fsm_id to " << fsm_id << std::endl;
    }

    if (arg_pair.first == "set_velocity") {
      std::vector<float> param = stringToFloatVector(arg_pair.second);
      auto param_size = param.size();
      float vx, vy, omega, duration;
      if (param_size == 3) {
        vx = param.at(0);
        vy = param.at(1);
        omega = param.at(2);
        duration = 1.f;
      } else if (param_size == 4) {
        vx = param.at(0);
        vy = param.at(1);
        omega = param.at(2);
        duration = param.at(3);
      } else {
        std::cerr << "Invalid param size for method SetVelocity: " << param_size << std::endl;
        return 1;
      }

      client.SetVelocity(vx, vy, omega, duration);
      std::cout << "set velocity to " << arg_pair.second << std::endl;
    }

    if (arg_pair.first == "damp") {
      client.Damp();
    }

    if (arg_pair.first == "start") {
      client.Start();
    }

    // if (arg_pair.first == "squat") {
    //   client.Squat();
    // }

    // if (arg_pair.first == "sit") {
    //   client.Sit();
    // }

    if (arg_pair.first == "stand_up") {
      client.StandUp();
    }

    if (arg_pair.first == "zero_torque") {
      client.ZeroTorque();
    }

    if (arg_pair.first == "stop_move") {
      client.StopMove();
    }

    if (arg_pair.first == "switch_move_mode") {
      bool flag;
      if (arg_pair.second == "true") {
        flag = true;
      } else if (arg_pair.second == "false") {
        flag = false;
      } else {
        std::cerr << "invalid argument: " << arg_pair.second << std::endl;
        return 1;
      }
      client.SwitchMoveMode(flag);
    }

    if (arg_pair.first == "move") {
      std::vector<float> param = stringToFloatVector(arg_pair.second);
      auto param_size = param.size();
      float vx, vy, omega;
      if (param_size == 3) {
        vx = param.at(0);
        vy = param.at(1);
        omega = param.at(2);
      } else {
        std::cerr << "Invalid param size for method SetVelocity: " << param_size << std::endl;
        return 1;
      }
      client.Move(vx, vy, omega);
    }

    if (arg_pair.first == "set_speed_mode") {
      int param = std::stoi(arg_pair.second);
      client.SetSpeedMode(param);
      std::cout << "set speed mode to " << arg_pair.second << std::endl;
    }

    std::cout << "Done!" << std::endl;
  }

  return 0;
}