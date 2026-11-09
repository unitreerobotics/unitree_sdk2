#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <thread>

#include <unitree/idl/hg/MotorCmd_.hpp>
#include <unitree/idl/hg/MotorState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

static const std::string WAIST_CMD_TOPIC = "rt/waistpitchcmd";
static const std::string WAIST_STATE_TOPIC = "rt/waistpitchstate";

using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

float current_q = 0.0;
bool get_current_q = false;

void WaistStateHandler(const void *message) {
  MotorState_ state = *(const MotorState_ *)message;
  get_current_q = true;
  current_q = state.q();
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: h2d_waist_example network_interface" << std::endl;
    return 0;
  }

  ChannelFactory::Instance()->Init(0, argv[1]);

  ChannelPublisherPtr<MotorCmd_> waist_publisher(new ChannelPublisher<MotorCmd_>(WAIST_CMD_TOPIC));
  waist_publisher->InitChannel();

  ChannelSubscriberPtr<MotorState_> waist_subscriber(new ChannelSubscriber<MotorState_>(WAIST_STATE_TOPIC));
  waist_subscriber->InitChannel(WaistStateHandler, 1);

  while(get_current_q == false){
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  MotorCmd_ command;
  command.mode() = 1;
  command.q() = 0.0F;
  command.dq() = 0.0F;
  command.tau() = 0.0F;
  command.kp() = 500.0F;
  command.kd() = 12.0F;

  waist_publisher->Write(command);

  while(std::abs(current_q - command.q()) > 0.05){
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  std::cout << "Reached q: " << command.q() << std::endl;

  for(;;){
    command.q() = 0.8F;
    waist_publisher->Write(command);
    while(std::abs(current_q - command.q()) > 0.05){
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Reached q: " << command.q() << std::endl;

    command.q() = 0.0F;
    waist_publisher->Write(command);
    while(std::abs(current_q - command.q()) > 0.05){
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Reached q: " << command.q() << std::endl;
  }
  return 0;
}
