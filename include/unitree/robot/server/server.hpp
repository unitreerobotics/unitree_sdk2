#ifndef __UT_ROBOT_SDK_SERVER_HPP__
#define __UT_ROBOT_SDK_SERVER_HPP__

#include <unitree/robot/server/server_base.hpp>
#include <unitree/robot/server/lease_server.hpp>

#define UT_ROBOT_SERVER_REG_API_HANDLER_NO_LEASE(apiId, handler) \
    UT_ROBOT_SERVER_REG_API_HANDLER(apiId, handler, false)

#define UT_ROBOT_SERVER_REG_API_HANDLER(apiId, handler, checkLease) \
    RegistHandler(apiId, std::bind(handler, this, std::placeholders::_1, std::placeholders::_2), checkLease)

namespace unitree
{
namespace robot
{
using RequestHandler = std::function<int32_t(const std::string& parameter,std::string& data)>;

class Server : public ServerBase
{
public:
    explicit Server(const std::string& name);
    virtual ~Server();

    virtual void Init() = 0;

    void StartLease(int64_t leaseTerm);
    void StartLease(float leaseTerm);

    const std::string& GetName();
    const std::string& GetApiVersion() const;

protected:
    void SetApiVersion(const std::string& version);

    void ServerRequestHandler(const RequestPtr& request);
    void RegistHandler(int32_t apiId, const RequestHandler& handler, bool checkLease = false);

    RequestHandler GetHandler(int32_t apiId, bool& ignoreLease) const;

    bool CheckLeaseDenied(int64_t leaseId);

private:
    bool mEnableLease;
    std::string mName;
    std::string mApiVersion;
    std::unordered_map<int32_t,std::pair<RequestHandler,bool>> mApiHandlerMap;
    LeaseServerPtr mLeaseServerPtr;
};

using ServerPtr = std::shared_ptr<Server>;

}
}

#endif//__UT_ROBOT_SDK_SERVER_HPP__
