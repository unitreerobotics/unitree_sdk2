#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unitree/idl/go2/LowCmd_.hpp>

namespace py = pybind11;

typedef struct
{
	uint8_t off; // off 0xA5
	std::array<uint8_t, 3> reserve;
} BmsCmd;

typedef struct
{
	uint8_t mode; // desired working mode
	float q;	  // desired angle (unit: radian)
	float dq;	  // desired velocity (unit: radian/second)
	float tau;	  // desired output torque (unit: N.m)
	float Kp;	  // desired position stiffness (unit: N.m/rad )
	float Kd;	  // desired velocity stiffness (unit: N.m/(rad/s) )
	std::array<uint32_t, 3> reserve;
} MotorCmd; // motor control

typedef struct
{
	std::array<uint8_t, 2> head;
	uint8_t levelFlag;
	uint8_t frameReserve;
		
	std::array<uint32_t, 2> SN;
	std::array<uint32_t, 2> version;
	uint16_t bandWidth;
	std::array<MotorCmd, 20> motorCmd;
	BmsCmd bms;
	std::array<uint8_t, 40> wirelessRemote;
	std::array<uint8_t, 12> led;
	std::array<uint8_t, 2> fan;
	uint8_t gpio;
	uint32_t reserve;
	
	uint32_t crc;
} LowCmd;           


uint32_t crc32_core(uint32_t* ptr, uint32_t len)
{
    unsigned int xbit = 0;
    unsigned int data = 0;
    unsigned int CRC32 = 0xFFFFFFFF;
    const unsigned int dwPolynomial = 0x04c11db7;

    for (unsigned int i = 0; i < len; i++)
    {
        xbit = 1 << 31;
        data = ptr[i];
        for (unsigned int bits = 0; bits < 32; bits++)
        {
            if (CRC32 & 0x80000000)
            {
                CRC32 <<= 1;
                CRC32 ^= dwPolynomial;
            }
            else
            {
                CRC32 <<= 1;
            }

            if (data & xbit)
                CRC32 ^= dwPolynomial;
            xbit >>= 1;
        }
    }

    return CRC32;
}

uint32_t get_crc(py::object msg)
{
    LowCmd raw{};

    std::array<uint8_t, 2> head = py::cast<std::array<uint8_t, 2>>(msg.attr("head"));

    memcpy(&raw.head, &head, 2);

    raw.levelFlag = py::cast<uint8_t>(msg.attr("level_flag"));
    raw.frameReserve = py::cast<uint8_t>(msg.attr("frame_reserve"));

    std::array<uint32_t, 2> SN = py::cast<std::array<uint32_t, 2>>(msg.attr("sn"));
    memcpy(&raw.SN, &SN, 8);

    std::array<uint32_t, 2> version = py::cast<std::array<uint32_t, 2>>(msg.attr("version"));
    memcpy(&raw.version, &version, 8);

    raw.bandWidth = py::cast<uint16_t>(msg.attr("bandwidth"));
    std::list<py::object> motor_cmds = py::cast<std::list<py::object>>(msg.attr("motor_cmd"));
    
    for (int i = 0; i < 20; i++)
    {
        py::object motor_cmd = motor_cmds.front();
        uint8_t mode = py::cast<uint8_t>(motor_cmd.attr("mode"));
        float q = py::cast<float>(motor_cmd.attr("q"));
        float dq = py::cast<float>(motor_cmd.attr("dq"));
        float tau = py::cast<float>(motor_cmd.attr("tau"));
        float Kp = py::cast<float>(motor_cmd.attr("kp"));
        float Kd = py::cast<float>(motor_cmd.attr("kd"));
        raw.motorCmd[i].mode = mode;
        raw.motorCmd[i].q = q;
        raw.motorCmd[i].dq = dq;
        raw.motorCmd[i].tau = tau;
        raw.motorCmd[i].Kp = Kp;
        raw.motorCmd[i].Kd = Kd;
        motor_cmds.pop_front();
    }

    BmsCmd bms;
    py::object bms_msg = msg.attr("bms_cmd");
    bms.off = py::cast<uint8_t>(bms_msg.attr("off"));
    bms.reserve = py::cast<std::array<uint8_t, 3>>(bms_msg.attr("reserve"));
    raw.bms.off = bms.off;
    memcpy(&raw.bms.reserve, &bms.reserve, bms.reserve.size()*sizeof(uint8_t));

    std::array<uint8_t, 40> wirelessRemote = py::cast<std::array<uint8_t, 40>>(msg.attr("wireless_remote"));
    memcpy(&raw.wirelessRemote, &wirelessRemote, wirelessRemote.size()*sizeof(uint8_t));
    std::array<uint8_t, 12> led = py::cast<std::array<uint8_t, 12>>(msg.attr("led"));
    memcpy(&raw.led, &led, led.size()*sizeof(uint8_t));
    std::array<uint8_t, 2> fan = py::cast<std::array<uint8_t, 2>>(msg.attr("fan"));
    memcpy(&raw.fan, &fan, fan.size()*sizeof(uint8_t));

    raw.gpio = py::cast<uint8_t>(msg.attr("gpio"));
    raw.reserve = py::cast<uint32_t>(msg.attr("reserve"));

    // printf("head: %x %x\n", raw.head[0], raw.head[1]);
    // printf("levelFlag: %x\n", raw.levelFlag);
    // printf("frameReserve: %x\n", raw.frameReserve);
    // printf("SN: %x %x\n", raw.SN[0], raw.SN[1]);
    // printf("version: %x %x\n", raw.version[0], raw.version[1]);
    // printf("bandWidth: %x\n", raw.bandWidth);
    // printf("motorCmd[0]: %x %f %f %f %f %f\n", raw.motorCmd[0].mode, raw.motorCmd[0].q, raw.motorCmd[0].dq, raw.motorCmd[0].tau, raw.motorCmd[0].Kp, raw.motorCmd[0].Kd);
    // printf("motorCmd[1]: %x %f %f %f %f %f\n", raw.motorCmd[1].mode, raw.motorCmd[1].q, raw.motorCmd[1].dq, raw.motorCmd[1].tau, raw.motorCmd[1].Kp, raw.motorCmd[1].Kd);
    // printf("motorCmd[2]: %x %f %f %f %f %f\n", raw.motorCmd[2].mode, raw.motorCmd[2].q, raw.motorCmd[2].dq, raw.motorCmd[2].tau, raw.motorCmd[2].Kp, raw.motorCmd[2].Kd);
    // printf("bms: %x %x %x %x\n", raw.bms.off, raw.bms.reserve[0], raw.bms.reserve[1], raw.bms.reserve[2]);
    // printf("wirelessRemote: %x %x %x %x\n", raw.wirelessRemote[0], raw.wirelessRemote[1], raw.wirelessRemote[2], raw.wirelessRemote[3]);
    // printf("led: %x %x %x %x\n", raw.led[0], raw.led[1], raw.led[2], raw.led[3]);
    // printf("fan: %x %x\n", raw.fan[0], raw.fan[1]);
    // printf("gpio: %x\n", raw.gpio);
    // printf("reserve: %x\n", raw.reserve);
    // printf("crc: %x\n", raw.crc);

    // for (int i = 0; i < (sizeof(LowCmd)>>2)-1; i++){
    //     printf("%x\n", ((uint32_t *)&raw)[i]);
    // }

    uint32_t crc = crc32_core((uint32_t *)&raw, (sizeof(LowCmd)>>2)-1);
    // printf("crc: %d, size: %d\n", crc, (sizeof(LowCmd)>>2)-1);
    return crc;
}

PYBIND11_MODULE(crc_module, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("crc32_core", &crc32_core, "A function which calculates the crc32 of a given array of uint32_t");
    m.def("get_crc", &get_crc, "A function which calculates the crc32 of a given LowCmd_ message");
}