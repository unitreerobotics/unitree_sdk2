#ifndef __ICEORYXVERSIONS__
#define __ICEORYXVERSIONS__

#define ICEORYX_VERSION_MAJOR    2
#define ICEORYX_VERSION_MINOR    0
#define ICEORYX_VERSION_PATCH    2
#define ICEORYX_VERSION_TWEAK    0

#define ICEORYX_LATEST_RELEASE_VERSION    "2.0.2"
#define ICEORYX_BUILDDATE                 "2023-08-31T07:36:50Z"
#define ICEORYX_SHA1                      ""

#include "iceoryx_posh/internal/log/posh_logging.hpp"

#define ICEORYX_PRINT_BUILDINFO()     iox::LogInfo() << "Built: " << ICEORYX_BUILDDATE;


#endif
