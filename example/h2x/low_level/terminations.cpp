#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/h2x/common/terminations.hpp>

using unitree::robot::ChannelFactory;
using unitree::robot::ChannelSubscriber;
using unitree_hg::msg::dds_::LowState_;

void PrintUsage() {
  std::cout
      << "Usage: h2x_termination [-n <network_interface>] [--network=<iface>]\n"
      << "\n"
      << "Subscribes rt/lowstate read-only and prints a warning when:\n"
      << "  - orientation exceeds 1 rad\n"
      << "  - no state message arrived for 1000 ms\n"
      << "\n"
      << "Options:\n"
      << "  -n, --network  dds interface, e.g. eth0\n"
      << "  -h, --help     show this help\n";
}

int main(int argc, char** argv) {
  // boost::program_options is not used here, it has no aarch64 build available.
  std::string network;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      PrintUsage();
      return 0;
    }
    if (arg.compare(0, 10, "--network=") == 0) {
      network = arg.substr(10);
    } else if (arg == "-n" && i + 1 < argc) {
      network = argv[++i];
    }
  }

  try {
    unitree::robot::ChannelFactory::Instance()->Init(0, network);
  } catch (const std::exception& e) {
    std::cerr << "dds init failed: " << e.what() << std::endl;
    return 1;
  }

  auto lowstate_subscriber =
      std::make_shared<ChannelSubscriber<LowState_>>("rt/lowstate");
  LowState_ lowstate;
  lowstate_subscriber->InitChannel([&lowstate](const void* message) {
    lowstate = *static_cast<const LowState_*>(message);
  });

  // Predicates and thresholds are the same as G1, see h2x/common/terminations.hpp.
  std::cout << "Checking terminations..." << std::endl;
  while (true) {
    if (unitree::robot::h2x::bad_orientation(lowstate, 1.0f)) {
      std::cout << "Bad orientation detected!" << std::endl;
    }
    if (unitree::robot::h2x::lost_connection(lowstate_subscriber, 1000)) {
      std::cout << "Lost connection!" << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}
