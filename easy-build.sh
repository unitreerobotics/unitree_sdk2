#!/usr/bin/env bash
# Build dist/ with Docker: ./easy-build.sh [--mirror HOST]
set -euo pipefail

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
readonly IMAGE=unitree-sdk2-builder

die() { printf '%s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage: ./easy-build.sh [--mirror HOST]

Requires Docker to be installed and working.

By default the base image is pulled from Docker Hub (ubuntu:20.04).
--mirror sets a registry mirror host, so the base image becomes
HOST/library/ubuntu:20.04.
EOF
}

main() {
    local mirror= build_cmd build_args

    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help) usage; return 0 ;;
            --mirror)
                [ $# -ge 2 ] || die "missing value for --mirror"
                mirror=$2
                shift
                ;;
            --mirror=*) mirror=${1#*=} ;;
            *) die "unknown option: $1 (use -h for help)" ;;
        esac
        shift
    done

    command -v docker >/dev/null 2>&1 || die "docker not found"
    docker info >/dev/null 2>&1 || die "docker is not available"

    mirror=${mirror#http://}
    mirror=${mirror#https://}
    mirror=${mirror%/}

    build_args=(build -t "$IMAGE" -f "$ROOT/.devcontainer/Dockerfile.devcontainer")
    if [ -n "$mirror" ]; then
        build_args+=(--build-arg "BASE_IMAGE=$mirror/library/ubuntu:20.04")
    fi
    build_args+=("$ROOT/.devcontainer")

    printf '==> Building image\n'
    docker "${build_args[@]}"

    case "$(uname -m)" in
        x86_64|amd64)  build_cmd='./build.sh all -c -d ' ;;
        aarch64|arm64) build_cmd='./build.sh sdk -c -d  && ./build.sh dist' ;;
        *)             die "unsupported architecture: $(uname -m)" ;;
    esac

    printf '==> Building dist\n'
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
        -v "$ROOT":/workspace -w /workspace "$IMAGE" sh -c "$build_cmd"

    [ -d "$ROOT/dist" ] || die "build finished but dist/ was not found"
    printf '==> Done: %s\n' "$ROOT/dist"
}

main "$@"
