#!/usr/bin/env bash
# Build Unitree SDK2 and stage it under buildout/output/<arch>/usr
# (without installing into the host filesystem).
#
#   scripts/build_sdk.sh                 # host architecture, Debug
#   scripts/build_sdk.sh -t aarch64 -j 8 # cross compile
#   scripts/build_sdk.sh -d -E           # Debug, library/headers only
#   scripts/build_sdk.sh -c              # remove this architecture's build and stage dirs first
set -euo pipefail

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

usage() {
    cat <<'EOF'
Usage: scripts/build_sdk.sh [options]

  -h, --help          Show this help
  -d, --debug         Debug build (default: Debug)
  -j, --jobs N        Parallel jobs (default: nproc)
  -t, --target ARCH   Target architecture: x86_64|amd64 | aarch64|arm64
                      (default: host architecture)
  -E, --no-examples   Do not build examples
  -c, --clean         Remove this architecture's build and stage dirs first

The toolchain and dependencies come from the current shell; cross builds
require <arch>-linux-gnu-g++. Using .devcontainer/ is recommended: it already
contains the host and aarch64 cross environments.
aarch64 cross builds skip yaml-cpp examples.

Output: buildout/output/<arch>/usr/{bin,include,lib,share}
EOF
}

die() { printf '%s\n' "$*" >&2; exit 1; }

normalize_arch() {
    case "$1" in
        amd64|x86_64)  printf 'x86_64' ;;
        arm64|aarch64) printf 'aarch64' ;;
        *)             return 1 ;;
    esac
}

run_dir() { printf '%s/buildout/run/%s' "$ROOT" "$1"; }
out_dir() { printf '%s/buildout/output/%s' "$ROOT" "$1"; }

check_inputs() {
    local arch=$1
    command -v cmake >/dev/null 2>&1 || die "cmake not found; install it first"
    [ -f "$ROOT/lib/$arch/libunitree_sdk2.a" ] \
        || die "prebuilt library for $arch not found: $ROOT/lib/$arch/libunitree_sdk2.a"
}

require_cross() {
    local arch=$1 cross="${arch}-linux-gnu"
    command -v "${cross}-g++" >/dev/null 2>&1 \
        || die "cross compiler ${cross}-g++ not found; install g++-${arch}-linux-gnu"
}

verify_stage() {
    local stage=$1
    [ -f "$stage/usr/lib/libunitree_sdk2.a" ] && [ -d "$stage/usr/include" ] \
        || die "staged output is incomplete: $stage/usr"
}

build() {
    local arch=$1 host_arch=$2 build_type=$3 examples=$4 jobs=$5 clean=$6
    local build_dir stage_dir cross expected_cxx cached_cxx
    build_dir=$(run_dir "$arch")
    stage_dir=$(out_dir "$arch")
    local args=(
        -S "$ROOT"
        -B "$build_dir"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DCMAKE_BUILD_TYPE="$build_type"
        -DCMAKE_INSTALL_PREFIX=/usr
        -DCMAKE_INSTALL_LIBDIR=lib
        -DBUILD_EXAMPLES="$examples"
    )

    if [ "$clean" -eq 1 ]; then
        printf '==> Cleaning (%s)\n' "$arch"
        rm -rf -- "$build_dir" "$stage_dir"
    fi

    if [ "$arch" != "$host_arch" ]; then
        require_cross "$arch"
        cross="${arch}-linux-gnu"
        expected_cxx=$(command -v "${cross}-g++")
        args+=(
            -DCMAKE_SYSTEM_NAME=Linux
            -DCMAKE_SYSTEM_PROCESSOR="$arch"
            -DCMAKE_LIBRARY_ARCHITECTURE="$cross"
            -DCMAKE_C_COMPILER="${cross}-gcc"
            -DCMAKE_CXX_COMPILER="${cross}-g++"
            # The yaml-cpp dev package in focal cannot coexist with the amd64
            # package; cross builds skip examples that depend on it so they
            # cannot accidentally link the host x86_64 library.
            -DCMAKE_DISABLE_FIND_PACKAGE_yaml-cpp=TRUE
        )
        # Debian multiarch directories are not in the cross compiler's default
        # search path, so point Boost at the matching directory explicitly.
        if [ -d "/usr/lib/$cross" ]; then
            args+=(-DBOOST_LIBRARYDIR="/usr/lib/$cross")
        fi
    else
        expected_cxx=$(command -v "${CXX:-c++}")
    fi

    # The build directory caches the source path and compiler. Reconfigure when
    # either changes (for example when the repository is mounted at /workspace),
    # otherwise CMake keeps using the stale cache.
    if [ -f "$build_dir/CMakeCache.txt" ]; then
        cached_home=$(sed -n 's/^CMAKE_HOME_DIRECTORY:[^=]*=//p' "$build_dir/CMakeCache.txt")
        if [ -n "$cached_home" ] && [ "$cached_home" != "$ROOT" ]; then
            printf '==> Source path changed; removing build directory: %s\n' "$build_dir"
            rm -rf -- "$build_dir"
        fi
    fi
    if [ -f "$build_dir/CMakeCache.txt" ] && [ -n "$expected_cxx" ]; then
        cached_cxx=$(sed -n 's/^CMAKE_CXX_COMPILER:[^=]*=//p' "$build_dir/CMakeCache.txt")
        if [ -n "$cached_cxx" ] \
            && [ "$(readlink -f -- "$cached_cxx" 2>/dev/null || printf '%s' "$cached_cxx")" \
                 != "$(readlink -f -- "$expected_cxx" 2>/dev/null || printf '%s' "$expected_cxx")" ]; then
            printf '==> Compiler changed; removing build directory: %s\n' "$build_dir"
            rm -rf -- "$build_dir"
        fi
    fi

    printf '==> Configuring (%s, %s, examples=%s)\n' "$arch" "$build_type" "$examples"
    printf '    Build directory: %s\n' "$build_dir"
    cmake "${args[@]}"

    printf '==> Building (-j %s)\n' "$jobs"
    cmake --build "$build_dir" --parallel "$jobs"

    rm -rf -- "$stage_dir"
    mkdir -p -- "$stage_dir"
    DESTDIR="$stage_dir" cmake --install "$build_dir"
    verify_stage "$stage_dir"

    printf '==> Staged: %s/usr\n' "$stage_dir"
}

main() {
    local arch target host_arch build_type=Debug examples=ON jobs clean=0
    jobs=$(nproc 2>/dev/null || echo 4)
    host_arch=$(normalize_arch "$(uname -m)") || host_arch=$(uname -m)

    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help) usage; return 0 ;;
            -d|--debug) build_type=Debug ;;
            -j|--jobs)
                [ $# -ge 2 ] || die "missing value for $1 (use -h for help)"
                jobs=$2
                shift
                ;;
            -t|--target)
                [ $# -ge 2 ] || die "missing value for $1 (use -h for help)"
                target=$2
                shift
                ;;
            -E|--no-examples) examples=OFF ;;
            -c|--clean) clean=1 ;;
            *) die "unknown option: $1 (use -h for help)" ;;
        esac
        shift
    done

    target=${target:-$host_arch}
    arch=$(normalize_arch "$target") || die "unsupported target architecture: $target"
    check_inputs "$arch"
    build "$arch" "$host_arch" "$build_type" "$examples" "$jobs" "$clean"
}

main "$@"
