#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unitree/idl/go2/LowCmd_.hpp>

namespace py = pybind11;

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
    // LowCmd raw;
    std::vector<uint32_t> head = py::cast<std::vector<uint32_t>>(msg.attr("head"));
    auto size = head.size();
    printf("head len: %ld, vals: %d %d\n", size, head[0], head[1]);
    return 0;
    
    // uint32_t crc = crc32_core((uint32_t *)&raw, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
    // return crc;
}

PYBIND11_MODULE(crc_module, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("crc32_core", &crc32_core, "A function which calculates the crc32 of a given array of uint32_t");
    m.def("get_crc", &get_crc, "A function which calculates the crc32 of a given LowCmd_ message");
}