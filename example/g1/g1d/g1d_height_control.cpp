#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#include <unitree/idl/ros2/Point32_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/g1/agv/g1_agv_client.hpp>

static const std::string G1D_HISPEED_STATE_TOPIC = "rt/hispeed_state";

using namespace unitree::robot;
using namespace geometry_msgs::msg::dds_;

volatile std::sig_atomic_t running = 1;
std::atomic<bool> height_received{false};
std::atomic<float> current_height{0.0F};
std::atomic<int64_t> last_height_time_ms{0};

int64_t NowMs() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

//Process Ctrl+C interrupt signal.
void SignalHandler(int) {
  running = 0;
}

void HispeedStateHandler(const void *message) {
  const Point32_ *hispeed_state = static_cast<const Point32_ *>(message);

  current_height.store(hispeed_state->y(), std::memory_order_relaxed);
  last_height_time_ms.store(NowMs(), std::memory_order_release);
  height_received.store(true, std::memory_order_release);
}

bool WaitForHeightData() {
  while (running != 0 && !height_received.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  return running != 0;
}

bool HeightDataTimeout() {
  const int64_t last_time = last_height_time_ms.load(std::memory_order_acquire);
  return last_time <= 0 || NowMs() - last_time > 500;
}

bool ParseFloat(const char *text, float *value) {
  char *end = nullptr;
  errno = 0;
  const float parsed = std::strtof(text, &end);
  if (text == end || *end != '\0' || errno == ERANGE || !std::isfinite(parsed)) {
    return false;
  }

  *value = parsed;
  return true;
}

void StopHeightAdjust(g1::AgvClient &agv_client) {
  const int32_t ret = agv_client.HeightAdjust(0.0F);
  if (ret != 0) {
    std::cout << std::endl << "[WARN] HeightAdjust stop command failed, ret: " << ret << std::endl;
  }
}

//Use PD controller to control the robot's height.
void height_control(g1::AgvClient &agv_client, float z) {
  constexpr float kP = 20.0F;
  constexpr float kD = 0.3F;
  constexpr float kMaxVz = 0.8F;
  constexpr float kArriveTolerance = 0.001F;
  constexpr int kArriveStableCycles = 20;
  constexpr auto kControlPeriod = std::chrono::milliseconds(50);

  if (!WaitForHeightData()) {
    StopHeightAdjust(agv_client);
    return;
  }

  float previous_error = z - current_height.load(std::memory_order_relaxed);
  int arrive_count = 0;
  auto previous_time = std::chrono::steady_clock::now();

  while (running != 0) {
    if (HeightDataTimeout()) {
      StopHeightAdjust(agv_client);
      std::cout << std::endl << "[ERROR] Height data timeout." << std::endl;
      return;
    }

    const auto now = std::chrono::steady_clock::now();
    float dt = std::chrono::duration<float>(now - previous_time).count();
    if (dt <= 0.0F) {
      dt = std::chrono::duration<float>(kControlPeriod).count();
    }
    previous_time = now;

    const float height = current_height.load(std::memory_order_relaxed);
    const float error = z - height;
    const float d_error = (error - previous_error) / dt;
    previous_error = error;

    float vz = kP * error + kD * d_error;
    vz = std::clamp(vz, -kMaxVz, kMaxVz);

    if (std::fabs(error) <= kArriveTolerance) {
      vz = 0.0F;
      ++arrive_count;
    } else {
      arrive_count = 0;
    }

    const int32_t ret = agv_client.HeightAdjust(vz);
    std::cout << "\rG1D column height: " << std::fixed << std::setprecision(4)
              << height << " m"
              << " | target: " << z << " m"
              << " | error: " << error << " m"
              << " | vz: " << vz << "    " << std::flush;

    if (ret != 0) {
      StopHeightAdjust(agv_client);
      std::cout << std::endl << "[ERROR] HeightAdjust failed, ret: " << ret << std::endl;
      return;
    }

    if (arrive_count >= kArriveStableCycles) {
      StopHeightAdjust(agv_client);
      std::cout << std::endl
                << "Target height reached: " << std::fixed << std::setprecision(4)
                << z << " m" << std::endl;
      return;
    }

    std::this_thread::sleep_for(kControlPeriod);
  }

  StopHeightAdjust(agv_client);
}

int main(int argc, char const *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: g1d_height_control network_interface target_height_m" << std::endl;
    return 0;
  }

  std::signal(SIGINT, SignalHandler);

  const std::string network_interface = argv[1];
  float target_height = 0.0F;
  if (!ParseFloat(argv[2], &target_height)) {
    std::cout << "Invalid target_height_m: " << argv[2] << std::endl;
    return 0;
  }

  ChannelFactory::Instance()->Init(0, network_interface);

  g1::AgvClient agv_client;
  agv_client.SetTimeout(3.0F);
  agv_client.Init();

  ChannelSubscriberPtr<Point32_> hispeed_state_subscriber(
      new ChannelSubscriber<Point32_>(G1D_HISPEED_STATE_TOPIC));
  hispeed_state_subscriber->InitChannel(HispeedStateHandler, 1);

  std::cout << "Subscribe topic: " << G1D_HISPEED_STATE_TOPIC << std::endl;
  std::cout << "Waiting for height data..." << std::endl;

  height_control(agv_client, target_height);

  std::cout << std::endl << "Exit g1d_height_control." << std::endl;
  return 0;
}
