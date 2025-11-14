#!/bin/bash
set -e

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    LIB_ARCH="x86_64"
    PLATFORM_TAG="linux_x86_64"
elif [ "$ARCH" = "aarch64" ]; then
    LIB_ARCH="aarch64"
    PLATFORM_TAG="linux_aarch64"
else
    echo "Unsupported architecture: $ARCH"
    exit 1
fi

echo "Building for $ARCH"

# Clean and build SDKs
mkdir -p build

# Build for Python 3.8 and 3.10
for PYTHON_VER in "3.8" "3.10"; do
    echo "Building Unitree SDK for Python $PYTHON_VER..."
    
    BUILD_DIR="build_py${PYTHON_VER//.}"
    mkdir -p $BUILD_DIR && cd $BUILD_DIR
    
    # Set Python-specific configuration
    if [ "$PYTHON_VER" = "3.10" ]; then
        # For Python 3.10 built from source
        cmake .. \
            -DBUILD_PYTHON_BINDING=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
            -DCMAKE_INSTALL_RPATH='$ORIGIN' \
            -DPYTHON_EXECUTABLE=/usr/local/bin/python$PYTHON_VER \
            -DPYTHON_INCLUDE_DIR=/usr/local/include/python$PYTHON_VER \
            -DPYTHON_LIBRARY=/usr/local/lib/libpython$PYTHON_VER.so
    else
        # For system Python 3.8
        cmake .. \
            -DBUILD_PYTHON_BINDING=ON \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
            -DCMAKE_INSTALL_RPATH='$ORIGIN' \
            -DPYTHON_EXECUTABLE=/usr/bin/python$PYTHON_VER
    fi
    
    make -j$(nproc)
    cd ..

    # Copy Python binding for this version
    find $BUILD_DIR -name "unitree_interface.cpython-*.so" -exec cp {} unitree_interface/ \;
    
    # Copy third-party libraries (only once)
    if [ "$PYTHON_VER" = "3.8" ]; then
        cp thirdparty/lib/$LIB_ARCH/*.so* unitree_interface/ 2>/dev/null || true
        
        # Create versioned symlinks for FastRTPS libraries
        cd unitree_interface
        if [ -f libfastrtps.so ]; then
            ln -sf libfastrtps.so libfastrtps.so.2.13
        fi
        if [ -f libfastcdr.so ]; then
            ln -sf libfastcdr.so libfastcdr.so.2
        fi
        cd ..
    fi

    # Build wheel for this Python version
    python$PYTHON_VER -m build --wheel

    # Get Python version tag
    PYTHON_TAG=$(python$PYTHON_VER -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    echo "Using Python tag: $PYTHON_TAG"

    # Rename wheel
    cd dist
    for wheel in *.whl; do
        if [[ "$wheel" == *"-py3-none-any.whl" ]]; then
            new_name="${wheel/-py3-none-any.whl/-$PYTHON_TAG-$PYTHON_TAG-$PLATFORM_TAG.whl}"
            mv "$wheel" "$new_name"
            echo "Built wheel: $new_name"
        fi
    done
    cd ..
done

# Fix RPATH for all .so files
for so in unitree_interface/*.so*; do
    patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true
done
