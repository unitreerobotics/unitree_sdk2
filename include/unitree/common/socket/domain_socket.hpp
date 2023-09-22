#ifndef __UT_DOMAIN_SOCKET_HPP__
#define __UT_DOMAIN_SOCKET_HPP__

#include <unitree/common/socket/socket.hpp>

namespace unitree
{
namespace common
{
class DomainSocket
{
public:
    DomainSocket(const std::string& sockFileName = "", bool stream = false);
    virtual ~DomainSocket();

    int32_t GetFd();

    void Close();

private:
    virtual void Init(bool stream);

protected:
    int32_t mFd;
    std::string mSockFileName;
};

class StreamDomainSocket : public DomainSocket
{
public:
    StreamDomainSocket(const std::string& sockFileName = "");

    void Listen(int32_t backlog = 10);
    int32_t Accept(std::string* peerFileName = NULL);

    void Connect(const std::string& serverFileName);

public:
    int64_t Send(const void* buf, int64_t len);
    int64_t Recv(void* buf, int64_t len);

    int64_t LoopSend(const std::string& buf, int64_t len);
    int64_t LoopRecv(std::string& buf, int64_t len);

    int64_t Send(int32_t sockfd, const void* buf, int64_t len);
    int64_t Recv(int32_t sockfd, void* buf, int64_t len);

    int64_t LoopSend(int32_t sockfd, const std::string& buf, int64_t len);
    int64_t LoopRecv(int32_t sockfd, std::string& buf, int64_t len);
    int64_t LoopRecv(int32_t sockfd, std::string& buf, int64_t len, int64_t waitTimeout);

    void CloseFd(int32_t sockfd);
};

class DgramDomainSocket : public DomainSocket
{
public:
    DgramDomainSocket(const std::string& sockFileName = "");

public:
    int64_t Send(const void* buf, int64_t len, const std::string& peerFileName);
    int64_t Recv(void* buf, int64_t len, std::string* peerFileName = NULL);
};

}
}

#endif//__UT_DOMAIN_SOCKET_HPP__
