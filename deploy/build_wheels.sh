#!/bin/bash
set -e

UNITREE_TAG=$(grep '^version = ' pyproject.toml | cut -d'"' -f2)
PYTHON_VERSIONS=${PYTHON_VERSIONS:-"3.8 3.10"}
ARCHITECTURES=${ARCHITECTURES:-"x86_64 aarch64"}

echo "Building wheels for Python versions: $PYTHON_VERSIONS"
echo "Building for architectures: $ARCHITECTURES"

for PYTHON_VER in $PYTHON_VERSIONS; do
    for ARCH in $ARCHITECTURES; do
        echo "Building Python $PYTHON_VER wheel for $ARCH..."
        PY_TAG=$(echo $PYTHON_VER | tr -d '.')
        
        docker buildx build --platform linux/$ARCH \
            --build-arg UNITREE_TAG=$UNITREE_TAG \
            --build-arg PYTHON_VER=$PYTHON_VER \
            --build-arg PY_TAG=$PY_TAG \
            --build-arg ARCH=$ARCH \
            -f - \
            --output type=local,dest=./dist \
            . << EOF
FROM ubuntu:20.04 AS builder
ARG PYTHON_VER
ARG PY_TAG
ARG ARCH
ARG UNITREE_TAG
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \\
    build-essential cmake git python3-pip \\
    python3-pybind11 pybind11-dev patchelf wget curl \\
    libssl-dev libffi-dev libbz2-dev libreadline-dev libsqlite3-dev \\
    libncurses5-dev libncursesw5-dev xz-utils tk-dev libgdbm-dev \\
    libc6-dev libnss3-dev zlib1g-dev && \\
    rm -rf /var/lib/apt/lists/*

# Install Python 3.8 (already available) or compile Python 3.10 from source
RUN if [ "\$PYTHON_VER" = "3.8" ]; then \\
        apt-get update && apt-get install -y python3.8 python3.8-dev python3.8-distutils && rm -rf /var/lib/apt/lists/*; \\
    else \\
        cd /tmp && \\
        wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \\
        tar xzf Python-3.10.12.tgz && \\
        cd Python-3.10.12 && \\
        ./configure --enable-optimizations --prefix=/usr/local && \\
        make -j\$(nproc) && \\
        make altinstall && \\
        ln -sf /usr/local/bin/python3.10 /usr/bin/python3.10 && \\
        cd / && rm -rf /tmp/Python-3.10.12*; \\
    fi

RUN python\$PYTHON_VER -m pip install --upgrade pip setuptools wheel build pybind11

WORKDIR /work
COPY . .

RUN rm -rf unitree_interface/*.so* build_py\$PY_TAG dist && \\
    mkdir -p build_py\$PY_TAG && cd build_py\$PY_TAG && \\
    cmake .. -DBUILD_PYTHON_BINDING=ON -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON -DCMAKE_INSTALL_RPATH='\$ORIGIN' \\
    -DPYTHON_EXECUTABLE=/usr/bin/python\$PYTHON_VER && \\
    make -j\$(nproc) && cd .. && \\
    find build_py\$PY_TAG -name "unitree_interface.cpython-*.so" -exec cp {} unitree_interface/ \\; && \\
    cp thirdparty/lib/\$ARCH/*.so* unitree_interface/ && \\
    patchelf --set-rpath '\$ORIGIN' unitree_interface/unitree_interface.cpython-*.so && \\
    python\$PYTHON_VER -m build --wheel && \\
    cd dist && \\
    for wheel in *.whl; do \\
        mv "\$wheel" "unitree_sdk2-\$UNITREE_TAG-cp\$PY_TAG-cp\$PY_TAG-linux_\$ARCH.whl"; \\
        break; \\
    done

FROM scratch AS output
COPY --from=builder /work/dist/*.whl /
EOF
    done
done

echo "All wheels built successfully in dist/"
