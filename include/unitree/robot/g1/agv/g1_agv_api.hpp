#ifndef __UT_ROBOT_G1_AGV_API_HPP__
#define __UT_ROBOT_G1_AGV_API_HPP__

#include <unitree/common/json/jsonize.hpp>

using namespace unitree::common;

namespace unitree
{
namespace robot
{
namespace g1
{
/*service name*/
const std::string AGV_SERVICE_NAME = "agv";

/*api version*/
const std::string AGV_API_VERSION = "1.0.0.1";

/*api id*/
const int32_t ROBOT_API_ID_AGV_MOVE = 1001;
const int32_t ROBOT_API_ID_AGV_HEIGHT_ADJUST = 1002;


class MoveParameter : public Jsonize
{
public:
    MoveParameter() :
        vx(0.0), vy(0.0), vyaw(0.0)
    {}

    ~MoveParameter()
    {}

public:
    void fromJson(JsonMap& json)
    {
        FromJson(json["vx"], vx);
        FromJson(json["vy"], vy);
        FromJson(json["vyaw"], vyaw);
    }

    void toJson(JsonMap& json) const
    {
        ToJson(vx, json["vx"]);
        ToJson(vy, json["vy"]);
        ToJson(vyaw, json["vyaw"]);
    }

public:
    float vx;
    float vy;
    float vyaw;
};

}   //namespace agv
}   //namespace robot
}   //namespace unitree

#endif  // __UT_ROBOT_G1_AGV_API_HPP__