# unitree_sdk2
Unitree robot SDK, version 2.

### Prebuild environment
* OS  (Ubuntu 20.04 LTS)
* CPU  (aarch64 and x86_64)
* Compiler  (gcc version 9.4.0)

### Environment Setup

Before building or running the SDK, ensure the following dependencies are installed:

- CMake (version 3.15 or higher)
- GCC (version 9.4.0)
- Make

You can install the required packages on Ubuntu 20.04 with:

```bash
apt-get update
apt-get install -y cmake build-essential libyaml-cpp-dev libeigen3-dev libboost-program-options-dev
```

### Build

`buildout/` is local staging (not delivered); `dist/` is the only deliverable:

```bash
./build.sh sdk          # host architecture: buildout/output/<arch>/usr
./build.sh dist         # assemble dist/unitree-sdk2-linux-<arch>/
./build.sh install      # sync dist/ to ~/.local/unitree/ (optional destination)
./build.sh all          # build and assemble x86_64, then aarch64
./build.sh clean        # remove buildout/ and dist/
```

`./build.sh sdk` builds Debug by default. `easy-build.sh` builds Debug without examples.

Single architecture, debug, and parallel builds:

```bash
./build.sh sdk -t aarch64 -j 8
./build.sh sdk -d
./build.sh sdk -E       # build libraries/headers only, no examples
./build.sh sdk -c       # remove this architecture's build and stage dirs first
./build.sh dist -t aarch64
```

Using `.devcontainer/` is recommended. The x86_64 image includes the host build
dependencies, `aarch64-linux-gnu-g++`, and the aarch64 Boost runtime:

```bash
# VS Code: Dev Containers: Reopen in Container
# Or from the command line:
docker build -t unitree-sdk2-dev -f .devcontainer/Dockerfile.devcontainer .devcontainer
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD":/workspace -w /workspace unitree-sdk2-dev ./build.sh all
```

For a one-command Docker Debug build without examples, use:

```bash
./easy-build.sh
```

On x86_64, the container can produce both x86_64 and aarch64 `dist` trees; an
aarch64 container builds only its native architecture. Outside a container,
install the apt packages above. aarch64 cross builds skip examples that depend
on yaml-cpp (the focal yaml-cpp dev package cannot coexist with the amd64
package); Boost examples build normally.

The `dist` layout is shown below. The whole directory can be moved with
`mv`/`rsync` and used elsewhere:

```
dist/unitree-sdk2-linux-<arch>/
├── bin/                                     example programs
├── include/                                 SDK headers
├── lib/                                     libunitree_sdk2.a, libddsc*.so, CMake package files
├── target-build.sh                          build template for user projects
└── unitree-sdk2-env.sh                      environment script
```

After assembly, `dist` is made read-only with `chmod -R a-w`; re-assembly or
`clean` restores write permission first.

Examples locate shared libraries through `$ORIGIN/../lib`; binaries do not
embed absolute source-tree or build-directory paths. `-ffile-prefix-map`
strips source paths from `__FILE__`/debug information.

### Environment script

```bash
source dist/unitree-sdk2-linux-<arch>/unitree-sdk2-env.sh
```

| Variable | Value / purpose |
| --- | --- |
| `UNITREE_SDK2_ROOT` | Artifact root (determined from the script's own location) |
| `UNITREE_SDK2_ARCH` | Architecture of this artifact, hard-coded at generation time |
| `UNITREE_SDK2_INCLUDE_DIR` | `<dist>/include` |
| `UNITREE_SDK2_LIB_DIR` | `<dist>/lib` |
| `UNITREE_SDK2_BIN_DIR` | `<dist>/bin` |
| `UNITREE_SDK2_CMAKE_DIR` | `<dist>/lib/cmake/unitree_sdk2` |
| `UT_SDK_PREFIX` | Same as `UNITREE_SDK2_ROOT` (legacy variable name) |
| `PATH` | Prepends `<dist>/bin` |

The script does not export `LD_LIBRARY_PATH`, `CPATH`, `LIBRARY_PATH`, or
`CMAKE_PREFIX_PATH`. Example programs run through RPATH; CMake consumers locate
the package with `-Dunitree_sdk2_DIR="$UNITREE_SDK2_CMAKE_DIR"`.

Sourcing it repeatedly does not append `PATH` twice. The script locates the
artifact from its own path, so renaming/moving the directory still works; only
`source` in bash/zsh is supported. After moving, source the script at the new
location; it overwrites any stale `UNITREE_SDK2_ROOT`. It prints the key
variables after sourcing; set `UNITREE_SDK2_ENV_QUIET=1` to suppress output.

### Use the SDK in your own CMake project

```bash
source dist/unitree-sdk2-linux-<arch>/unitree-sdk2-env.sh
cmake -S . -B build -Dunitree_sdk2_DIR="$UNITREE_SDK2_CMAKE_DIR"
```

```cmake
find_package(unitree_sdk2 REQUIRED)
target_link_libraries(your_target PRIVATE unitree_sdk2)
```

`unitree_sdk2_DIR` is the CMake package variable and is case-sensitive;
`UNITREE_SDK2_DIR` is not recognized.

The SDK does not provide a toolchain file or generate gcc/g++ wrappers.
`unitree_sdk2_DIR` only locates the package and does not override the caller's
`CMAKE_SYSROOT`, `CMAKE_PREFIX_PATH`, or compiler settings. When
cross-compiling, pass your own toolchain together with it:

```bash
cmake -S . -B build \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/your-toolchain.cmake \
    -Dunitree_sdk2_DIR="$UNITREE_SDK2_CMAKE_DIR"
```

Link libraries outside the sysroot directly; do not rely on `LIBRARY_PATH`
(the cross gcc discards paths outside the sysroot):

```cmake
link_directories(/opt/ext/lib)
target_link_libraries(your_target PRIVATE ext)
```

When running your own build, prefer setting an RPATH pointing to `<dist>/lib`;
alternatively set `LD_LIBRARY_PATH=<dist>/lib` for a single run command. Do not
prepend `LD_LIBRARY_PATH` globally: it can shadow system or other versions of
`libddsc*` and affects every process in that shell.

### Build options

```text
./build.sh sdk  [-d] [-j N] [-t ARCH] [-E] [-c]
./build.sh dist [-t ARCH]
./build.sh all  [-j N]
./build.sh clean
```

`all` requires the aarch64 cross compiler `aarch64-linux-gnu-g++` (included in
the x86_64 devcontainer).

### Verify relocatability

```bash
# The artifact still runs after being moved
mv dist/unitree-sdk2-linux-x86_64 /tmp/sdk2-moved
source /tmp/sdk2-moved/unitree-sdk2-env.sh
readelf -d /tmp/sdk2-moved/bin/h2_dual_arm_example | grep -E 'RPATH|RUNPATH'
# Expected: Library rpath: [$ORIGIN/../lib]

# The artifact should not contain the repository path
grep -rl '/path/to/your/repo' /tmp/sdk2-moved || echo "clean"
```

### Notice
For more reference information, please go to [Unitree Document Center](https://support.unitree.com/home/zh/developer).
