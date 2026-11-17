#!/usr/bin/env bash
set -euo pipefail

: "${UNITREE_SDK2_CMAKE_DIR:?source unitree-sdk2-env.sh}"

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}" \
    -Dunitree_sdk2_DIR="$UNITREE_SDK2_CMAKE_DIR"
cmake --build build --parallel "${JOBS:-$(nproc)}" --target "${1:-all}"
