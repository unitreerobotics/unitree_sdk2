cmake_minimum_required(VERSION 3.5)
project(unitree_sdk2)

## Project Options
option(BUILD_EXAMPLES "Build examples" ON)

## Set compiler to use c++ 17 features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Chosse build type
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

## Use GNUInstallDirs to install libraries into correct locations on all platforms.
include(GNUInstallDirs)

## Put all binary files into /bin and libraries into /lib
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

## Check system architecture
message(STATUS "Detected system architecture: ${CMAKE_SYSTEM_PROCESSOR}")

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(Platform "arm64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "armv7l")
    set(Platform "arm32")
else()
    set(Platform "amd64")
endif()

## Import thirdparty libraries
add_subdirectory(thirdparty)

## Import Unitree SDK2 library
set(UNITREE_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/lib/${CMAKE_SYSTEM_PROCESSOR})

find_library(UNITREE_SDK_LIB unitree_sdk2 PATHS ${UNITREE_SDK_PATH})

if(NOT UNITREE_SDK_LIB)
    message(FATAL_ERROR "Unitree SDK library not found")
else()
    message(STATUS "Unitree SDK library found at: ${UNITREE_SDK_LIB}")
endif()

add_library(unitree_sdk INTERFACE)
target_link_libraries(unitree_sdk INTERFACE
        ddsc
        ddscxx
        ${UNITREE_SDK_LIB})
target_include_directories(unitree_sdk INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

if(BUILD_EXAMPLES)
    add_subdirectory(example)
endif()



