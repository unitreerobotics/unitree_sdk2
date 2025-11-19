#!/bin/bash
set -e

ARCH=$1
PY_VER=$2

if [ -z "$ARCH" ] || [ -z "$PY_VER" ]; then
    echo "Usage: $0 <x86_64|aarch64> <3.10|3.11>"
    exit 1
fi

WHEEL="unitree_sdk2-0.1.1-cp${PY_VER//./}-cp${PY_VER//./}-linux_${ARCH}.whl"

echo "Testing $WHEEL on $ARCH with Python $PY_VER"

docker run --rm --platform linux/$ARCH \
    -v "$(pwd)/dist:/wheels:ro" \
    -v "$(pwd)/test_wheels.py:/test.py:ro" \
    python:${PY_VER}-slim \
    bash -c "pip install -q /wheels/$WHEEL && python /test.py"
