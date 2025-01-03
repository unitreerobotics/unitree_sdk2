
#include <cmath>  // for generate wave
#include <iostream>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

const int SAMPLE_RATE = 16000;   // 16K sample rate
const int DURATION = 5;          // time S
const float FREQUENCY = 440.0f;  // HZ

void generateSineWave(std::vector<uint8_t> &audioData) {
  int numSamples = SAMPLE_RATE * DURATION;

  for (int i = 0; i < numSamples; ++i) {
    float time = i / float(SAMPLE_RATE);
    float value = sin(2 * M_PI * FREQUENCY * time);
    int16_t int16Value = static_cast<int16_t>(value * 32767);
    uint8_t lowByte = static_cast<uint8_t>(int16Value & 0xFF);
    uint8_t highByte = static_cast<uint8_t>((int16Value >> 8) & 0xFF);
    audioData.push_back(lowByte);
    audioData.push_back(highByte);
  }
}

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: audio_client_example [NetWorkInterface(eth0)]"
              << std::endl;
    exit(0);
  }
  /*
   * Initilaize ChannelFactory
   */
  unitree::robot::ChannelFactory::Instance()->Init(0, argv[1]);
  unitree::robot::g1::AudioClient client;
  client.Init();
  client.SetTimeout(10.0f);

  /*TTS Example*/
  int32_t ret;
  ret = client.TtsMaker("你好。我是宇树科技的机器人G1。例程启动成功",
                        0);  // Auto play
  std::cout << "TtsMaker ret:" << ret << std::endl;
  unitree::common::Sleep(5);

  /*Volume Example*/
  uint8_t volume;
  ret = client.GetVolume(volume);
  std::cout << "GetVolume ret:" << ret
            << "  volume = " << std::to_string(volume) << std::endl;
  ret = client.SetVolume(60);
  std::cout << "SetVolume to 60% , ret:" << ret << std::endl;

  /*Audio Play Example*/
  std::vector<uint8_t> pcm;
  generateSineWave(pcm);
  client.PlayStream(
      "example", std::to_string(unitree::common::GetCurrentTimeMillisecond()),
      pcm);
  std::cout << "start play" << std::endl;
  unitree::common::Sleep(3);
  std::cout << "stop play" << std::endl;
  ret = client.PlayStop("example");

  /*LED Control Example*/
  client.LedControl(0, 255, 0);
  unitree::common::Sleep(1);
  client.LedControl(0, 0, 0);
  unitree::common::Sleep(1);
  client.LedControl(0, 0, 255);

  std::cout << "AudioClient test finish!" << std::endl;
  return 0;
}