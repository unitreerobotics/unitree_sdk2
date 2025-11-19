#!/bin/bash
set -e

echo "Testing all Python wheels..."
echo

for ARCH in x86_64 aarch64; do
    for PY_VER in 3.10 3.11; do
        ./test_docker.sh $ARCH $PY_VER
        echo
    done
done

echo "All tests completed successfully!"
