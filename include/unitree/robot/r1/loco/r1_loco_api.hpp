#pragma once

#include <unitree/common/json/jsonize.hpp>
#include <variant>

namespace unitree {
namespace robot {
    namespace r1 {

        const std::string LOCO_SERVICE_NAME = "sport";

        /*api version*/
        const std::string LOCO_API_VERSION = "1.0.0.0";

        /*api id*/
        const int32_t ROBOT_API_ID_LOCO_GET_FSM_ID         = 7001;
        const int32_t ROBOT_API_ID_LOCO_GET_FSM_MODE       = 7002;

        const int32_t ROBOT_API_ID_LOCO_SET_FSM_ID         = 7101;
        const int32_t ROBOT_API_ID_LOCO_SET_VELOCITY       = 7105;
        const int32_t ROBOT_API_ID_LOCO_SET_SPEED_MODE     = 7107;

        using LocoCmd = std::map< std::string, std::variant< int, float, std::vector< float > > >;

        class JsonizeDataVecFloat : public common::Jsonize {
        public:
            JsonizeDataVecFloat() {}
            ~JsonizeDataVecFloat() {}

            void fromJson( common::JsonMap& json ) {
                common::FromJson( json[ "data" ], data );
            }

            void toJson( common::JsonMap& json ) const {
                common::ToJson( data, json[ "data" ] );
            }

            std::vector< float > data;
        };

        class JsonizeVelocityCommand : public common::Jsonize {
        public:
            JsonizeVelocityCommand() {}
            ~JsonizeVelocityCommand() {}

            void fromJson( common::JsonMap& json ) {
                common::FromJson( json[ "velocity" ], velocity );
                common::FromJson( json[ "duration" ], duration );
            }

            void toJson( common::JsonMap& json ) const {
                common::ToJson( velocity, json[ "velocity" ] );
                common::ToJson( duration, json[ "duration" ] );
            }

            std::vector< float > velocity;
            float duration;
        };
    }  // namespace r1
}  // namespace robot
}  // namespace unitree
