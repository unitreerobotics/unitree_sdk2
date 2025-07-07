cmake_minimum_required(VERSION 3.12)
project(g1_interface_python)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(pybind11 REQUIRED)

# Use the unitree_sdk2 target from parent project
# This automatically provides all necessary include paths and libraries
if(NOT TARGET unitree_sdk2)
    message(FATAL_ERROR "unitree_sdk2 target not found. Make sure this is built from the parent project.")
endif()

# Add source files
set(SOURCES
    g1_interface.cpp
    python_bindings.cpp
)

# Create pybind11 module
pybind11_add_module(g1_interface ${SOURCES})

# Compiler-specific options
target_compile_definitions(g1_interface PRIVATE VERSION_INFO=${PROJECT_VERSION})

# Link with unitree_sdk2 which provides all necessary dependencies
target_link_libraries(g1_interface PRIVATE unitree_sdk2)

# Enable compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(g1_interface PRIVATE -Wall -Wextra)
endif()

# Optional: Generate stub files for type hints
option(GENERATE_STUBS "Generate Python stub files" ON)

if(GENERATE_STUBS)
    # Add custom target to generate stub files
    add_custom_target(generate_stubs
        COMMAND ${CMAKE_COMMAND} -E echo "Generating Python stub files..."
        COMMAND python3 -c "
import sys
sys.path.insert(0, '${CMAKE_CURRENT_BINARY_DIR}')
try:
    import g1_interface
    import pybind11_stubgen
    pybind11_stubgen.main(['-o', '${CMAKE_CURRENT_BINARY_DIR}', 'g1_interface'])
    print('Stub files generated successfully')
except ImportError as e:
    print(f'Warning: Could not generate stub files: {e}')
    print('Install pybind11-stubgen: pip install pybind11-stubgen')
except Exception as e:
    print(f'Error generating stubs: {e}')
"
        DEPENDS g1_interface
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating Python stub files"
    )
endif()

# Installation
install(TARGETS g1_interface DESTINATION .)

# Print configuration summary
message(STATUS "=== G1 Interface Python Binding Configuration ===")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "Using unitree_sdk2 target from parent project")
message(STATUS "GENERATE_STUBS: ${GENERATE_STUBS}")
message(STATUS "===================================================") 