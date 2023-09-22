#ifndef __UT_SOCKET_HPP__
#define __UT_SOCKET_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace common
{
class SocketHelper
{
public:
    static SocketHelper* Instance()
    {
        static SocketHelper inst;
        return &inst;
    }

    int32_t Socket(int32_t domain, int32_t type, int32_t protocol);

    void Bind(int32_t sockfd, const struct sockaddr *addr, socklen_t addrlen);

    void Listen(int32_t sockfd, int32_t backlog);
    int32_t Accept(int32_t sockfd, struct sockaddr *addr, socklen_t *addrlen);
    int32_t Accept4(int32_t sockfd, struct sockaddr *addr, socklen_t *addrlen, int32_t flags);

    void Connect(int32_t sockfd, const struct sockaddr *addr, socklen_t addrlen);
    void Close(int32_t sockfd);

    void SetFlag(int32_t sockfd, int32_t flags);

    int64_t Send(int32_t sockfd, const void* buf, int64_t len);
    int64_t Send(int32_t sockfd, const void* buf, int64_t len, const struct sockaddr* addr, socklen_t addrlen);

    int64_t Recv(int32_t sockfd, void* buf, int64_t len);
    int64_t Recv(int32_t sockfd, void* buf, int64_t len, struct sockaddr* addr, socklen_t* addrlen);

    int64_t LoopSend(int32_t sockfd, const std::string& buf, int64_t len);
    int64_t LoopRecv(int32_t sockfd, std::string& buf, int64_t len);

    void WaitRecv(int32_t sockfd, int64_t waitTimeout);
    int64_t LoopRecv(int32_t sockfd, std::string& buf, int64_t len, int64_t waitTimeout);

private:
    SocketHelper();
};

}
}

#endif//__UT_SOCKET_HPP__
