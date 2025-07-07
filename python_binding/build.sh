#!/bin/bash

# G1 Interface Python Binding Build Script

set -e  # Exit on any error

echo "=== G1 Interface Python Binding Build Script ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}This script should not be run as root${NC}"
   exit 1
fi

# Default values
BUILD_TYPE="Release"
CLEAN=false
UNITREE_SDK_PATH="$(cd .. && pwd)"  # Use parent directory as default (where the main SDK is)
PYTHON_VERSION=""
INSTALL_DEPS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        --sdk-path)
            UNITREE_SDK_PATH="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --install-deps)
            INSTALL_DEPS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --debug           Build in debug mode"
            echo "  -c, --clean           Clean build directory first"
            echo "  --sdk-path PATH       Path to Unitree SDK (default: /opt/unitree/sdk2)"
            echo "  --python-version VER  Python version to use (e.g., 3.8, 3.9)"
            echo "  --install-deps        Install Python dependencies"
            echo "  -h, --help            Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "Build configuration:"
echo "  Build type: $BUILD_TYPE"
echo "  Unitree SDK path: $UNITREE_SDK_PATH"
echo "  Clean build: $CLEAN"

# Install Python dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install pybind11 pybind11-stubgen numpy
fi

# Check if Unitree SDK exists
if [ ! -d "$UNITREE_SDK_PATH" ]; then
    echo -e "${RED}Unitree SDK not found at: $UNITREE_SDK_PATH${NC}"
    echo "Please install Unitree SDK or specify correct path with --sdk-path"
    exit 1
fi

# Check for required dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check for pybind11
python -c "import pybind11" 2>/dev/null || {
    echo -e "${RED}pybind11 not found. Install with: pip install pybind11${NC}"
    exit 1
}

# Check for cmake
command -v cmake >/dev/null 2>&1 || {
    echo -e "${RED}cmake not found. Please install cmake${NC}"
    exit 1
}

# Check for make
command -v make >/dev/null 2>&1 || {
    echo -e "${RED}make not found. Please install build-essential${NC}"
    exit 1
}

# Create and enter build directory
BUILD_DIR="build"
if [ "$CLEAN" = true ] && [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run cmake
echo -e "${YELLOW}Running cmake...${NC}"
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DUNITREE_SDK_PATH="$UNITREE_SDK_PATH"
)

if [ -n "$PYTHON_VERSION" ]; then
    CMAKE_ARGS+=(-DPYTHON_VERSION="$PYTHON_VERSION")
fi

cmake "${CMAKE_ARGS[@]}" ..

# Build
echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

# Generate stub files if possible
echo -e "${YELLOW}Attempting to generate stub files...${NC}"
make generate_stubs || echo -e "${YELLOW}Warning: Could not generate stub files${NC}"

# Copy stub files to parent directory if they exist
if [ -f "g1_interface/g1_interface.pyi" ]; then
    cp g1_interface/g1_interface.pyi ../g1_interface_generated.pyi
    echo -e "${GREEN}Generated stub file copied to g1_interface_generated.pyi${NC}"
fi

# Copy compiled module to parent directory for easy testing
if [ -f "g1_interface*.so" ]; then
    cp g1_interface*.so ../
    echo -e "${GREEN}Compiled module copied to parent directory${NC}"
fi

cd ..

echo -e "${GREEN}Build completed successfully!${NC}"
echo ""
echo "Usage:"
echo "  1. Copy the compiled .so file to your Python path"
echo "  2. Use the g1_interface module in Python:"
echo "     import g1_interface"
echo "     robot = g1_interface.G1Interface('eth0')"
echo ""
echo "Files created:"
echo "  - build/g1_interface*.so (compiled Python module)"
if [ -f "g1_interface_generated.pyi" ]; then
    echo "  - g1_interface_generated.pyi (type hints)"
fi
echo ""
echo "Example usage:"
echo "  python3 example_ankle_swing.py eth0" 