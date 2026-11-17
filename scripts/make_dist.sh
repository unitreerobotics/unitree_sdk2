#!/usr/bin/env bash
# Assemble buildout/output/<arch>/usr into dist/unitree-sdk2-linux-<arch>/.
#
#   scripts/make_dist.sh                 # host architecture
#   scripts/make_dist.sh -t aarch64      # cross build
set -euo pipefail

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

usage() {
    cat <<'EOF'
Usage: scripts/make_dist.sh [options]

  -h, --help          Show this help
  -t, --target ARCH   Target architecture: x86_64|amd64 | aarch64|arm64
                      (default: host architecture)

Input:  buildout/output/<arch>/usr
Output: dist/unitree-sdk2-linux-<arch>/
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

stage_dir() { printf '%s/buildout/output/%s/usr' "$ROOT" "$1"; }
dist_dir()  { printf '%s/dist/unitree-sdk2-linux-%s' "$ROOT" "$1"; }

install_env() {
    local dist=$1 arch=$2
    sed "s/__ARCH__/$arch/g" "$ROOT/config/unitree-sdk2-env.sh.in" > "$dist/unitree-sdk2-env.sh"
    chmod +x "$dist/unitree-sdk2-env.sh"
}

install_target_build() {
    local dist=$1
    cp "$ROOT/config/target-build.sh" "$dist/target-build.sh"
    chmod +x "$dist/target-build.sh"
}

check_env() {
    local dist=$1 arch=$2
    env -u LD_LIBRARY_PATH -u CMAKE_PREFIX_PATH -u CPATH -u LIBRARY_PATH \
        bash -c 'set -e
                 . "$1" >/dev/null
                 [ "${UNITREE_SDK2_ARCH}" = "$2" ]
                 [ -d "${UNITREE_SDK2_CMAKE_DIR}" ]
                 [ -z "${LD_LIBRARY_PATH:-}" ]
                 [ -z "${CMAKE_PREFIX_PATH:-}" ]' \
        _ "$dist/unitree-sdk2-env.sh" "$arch"
}

check_rpath() {
    local dist=$1 example
    command -v readelf >/dev/null 2>&1 || return 0
    [ -d "$dist/bin" ] || return 0
    example=$(find "$dist/bin" -maxdepth 1 -type f -perm -u+x -print -quit 2>/dev/null)
    [ -n "$example" ] || return 0
    readelf -d "$example" | grep -q 'RPATH.*\$ORIGIN/../lib' \
        || die "example RPATH is not \$ORIGIN/../lib: $example"
}

check_repo_paths() {
    local dist=$1 matches
    matches=$(grep -rIlF -- "$ROOT" "$dist" || true)
    [ -z "$matches" ] || {
        printf '%s\n' "$matches" | head -n 5 >&2
        die "output still contains the build repository path: $ROOT"
    }
}

remove_dist() {
    local dist=$1
    [ -e "$dist" ] || return 0
    chmod -R u+w -- "$dist"
    rm -rf -- "$dist"
}

lock_dist() {
    chmod -R a-w -- "$1"
}

assemble() {
    local arch=$1 stage dist
    stage=$(stage_dir "$arch")
    dist=$(dist_dir "$arch")
    [ -d "$stage" ] || die "missing staged output: $stage
Run this first: ./build.sh sdk -t $arch"

    printf '==> Assembling %s\n' "$dist"
    remove_dist "$dist"
    mkdir -p -- "$dist"
    cp -a "$stage/." "$dist/"

    install_env "$dist" "$arch"
    install_target_build "$dist"
    check_env "$dist" "$arch"
    check_rpath "$dist"
    check_repo_paths "$dist"
    lock_dist "$dist"

    printf '==> Done: %s\n' "$dist"
}

main() {
    local arch target
    while [ $# -gt 0 ]; do
        case "$1" in
            -h|--help) usage; return 0 ;;
            -t|--target)
                [ $# -ge 2 ] || die "missing value for $1 (use -h for help)"
                arch=$2
                shift
                ;;
            *) die "unknown option: $1 (use -h for help)" ;;
        esac
        shift
    done
    target=${arch:-$(uname -m)}
    arch=$(normalize_arch "$target") || die "unsupported target architecture: $target"
    assemble "$arch"
}

main "$@"
