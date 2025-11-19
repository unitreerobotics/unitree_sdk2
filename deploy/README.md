# Wheel Build & Test Scripts

## Build Wheels

Build Python wheels for specific versions and architectures:

```bash
# Build for Python 3.11, both architectures
PYTHON_VERSIONS="3.11" ./build_wheels.sh

# Build for specific architecture
PYTHON_VERSIONS="3.11" ARCHITECTURES="x86_64" ./build_wheels.sh

# Build multiple versions
PYTHON_VERSIONS="3.10 3.11" ./build_wheels.sh
```

Wheels are output to `dist/` directory.

## Test Wheels

Test a specific wheel:
```bash
./test_docker.sh x86_64 3.11
./test_docker.sh aarch64 3.10
```

Test all wheels:
```bash
./run_tests.sh
```

## Requirements

- Docker with buildx support
- Wheels must be in `../dist/` directory
