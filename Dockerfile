FROM ubuntu:20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

RUN apt-get install -y \
    build-essential \
    cmake \
    git \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-pybind11 \
    pybind11-dev \
    patchelf \
    lsb-release \
    wget \
    curl \
    libssl-dev \
    ca-certificates \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Build Python 3.10 from source with development headers
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar xzf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-shared --prefix=/usr/local && \
    make -j$(nproc) && \
    make altinstall && \
    ldconfig && \
    cd / && rm -rf /tmp/Python-3.10.12*

# Install pip for each Python version
RUN python3.8 -m pip install --upgrade pip setuptools wheel build pybind11 pybind11-stubgen
RUN python3.10 -m pip install --upgrade pip setuptools wheel build pybind11 pybind11-stubgen

WORKDIR /work
COPY . .

ARG CONFIG_FILE=pyproject.toml
ARG UNITREE_TAG
RUN if [ "$CONFIG_FILE" != "pyproject.toml" ]; then cp $CONFIG_FILE pyproject.toml; fi

# Build for multiple Python versions
RUN ./build.sh

FROM scratch AS output
COPY --from=builder /work/dist/*.whl /
