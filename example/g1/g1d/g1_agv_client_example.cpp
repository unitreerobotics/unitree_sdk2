#include <unitree/robot/g1/agv/g1_agv_client.hpp>
#include <iostream>
#include <math.h>
#include <unistd.h>

extern "C" {
    float sinf(float x);
}

int main()
{
    /*
     * Initilaize ChannelFactory
     */
    unitree::robot::ChannelFactory::Instance()->Init(0);
    unitree::robot::g1::AgvClient ac;

    ac.SetTimeout(3.0f);
    ac.Init();

    //Test Api - Periodic motion
    int cycle_count = 0;
    const float cycle_period = 40.0f;  // 40步完成一个周期
    const float vx_amplitude = 0.3f;
    const float vyaw_amplitude = 0.3f;
    const float height_amplitude = 1.0f;
    
    while (true)
    {
        // 使用正弦函数实现周期运动
        float time_phase = (cycle_count % (int)cycle_period) / cycle_period * 2.0f * M_PI;
        
        // 速度周期运动: vx和vyaw使用正弦波
        float vx = vx_amplitude * sinf(time_phase);
        float vyaw = vyaw_amplitude * sinf(time_phase);
        
        int32_t ret = ac.Move(vx, 0.0f, vyaw);
        std::cout << "Call Move vx:" << vx << " vyaw:" << vyaw << " ret:" << ret << std::endl;

        usleep(50000);  // 50ms
        
        // 高度调节周期运动: 使用正弦波，周期与速度一致
        float height = height_amplitude * sinf(time_phase);
        
        int32_t ret2 = ac.HeightAdjust(height);
        std::cout << "Call HeightAdjust height:" << height << " ret:" << ret2 << std::endl;

        cycle_count++;
        usleep(50000);  // 50ms
    }

    return 0;
}
