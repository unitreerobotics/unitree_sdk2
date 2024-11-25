#ifndef __UT_LOG_BUFFER_HPP__
#define __UT_LOG_BUFFER_HPP__

#include <unitree/common/log/log_decl.hpp>

namespace unitree
{
namespace common
{
class LogBuffer
{
public:
    explicit LogBuffer();
    virtual ~LogBuffer();

    bool Append(const std::string& s);
    bool Get(std::string& s);
    bool Empty();

protected:
    std::string mData;
};

typedef std::shared_ptr<LogBuffer> LogBufferPtr;

class LogBlockBuffer : public LogBuffer
{
public:
    explicit LogBlockBuffer();
    ~LogBlockBuffer();

    bool Append(const std::string& s);
    bool Get(std::string& s);
    bool Empty();

private:
    Mutex mLock;
};

typedef std::shared_ptr<LogBlockBuffer> LogBlockBufferPtr;
}
}

#endif//__UT_LOG_BUFFER_HPP__
