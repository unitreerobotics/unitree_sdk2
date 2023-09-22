#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>

#define TOPIC_WIRELESSCONTROLLER "rt/wirelesscontroller"

using namespace unitree::robot;

int main(int argc, char** argv)
{
    std::string iface;
    if (argc > 1)
    {
        iface = argv[1];
    }

    ChannelFactory::Instance()->Init(0, iface);

    ChannelPublisher<unitree_go::msg::dds_::WirelessController_> publisher(TOPIC_WIRELESSCONTROLLER);
    publisher.InitChannel();

    while (true)
    {
        unitree_go::msg::dds_::WirelessController_ msg;
        msg.rx() = 0.2;

        /*
         * write message
         */
        publisher.Write(msg);

        sleep(1);
    }

    return 0;
}
