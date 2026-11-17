#!/usr/bin/env bash
# Unitree SDK2 build entry point: task dispatch only; details live in scripts/.
#
#   ./build.sh sdk              # build and stage to buildout/output/<arch>/usr
#   ./build.sh dist             # assemble dist/unitree-sdk2-linux-<arch>/
#   ./build.sh install [dir]    # sync dist/ to a directory (default ~/.local/unitree)
#   ./build.sh all              # build and assemble x86_64 and aarch64
#   ./build.sh clean            # remove buildout/ and dist/
set -euo pipefail

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

usage() {
    cat <<'EOF'
Usage: ./build.sh <task> [options...]

Tasks:
  sdk     Build and stage (options are passed to scripts/build_sdk.sh)
  dist    Assemble dist/ (options are passed to scripts/make_dist.sh)
  install Sync dist/ to a destination directory (default ~/.local/unitree)
  all     Build and assemble x86_64 and aarch64
  clean   Remove buildout/ and dist/

Examples:
  ./build.sh sdk
  ./build.sh sdk -t aarch64 -j 8
  ./build.sh sdk -c            # remove this architecture's build and stage dirs first
  ./build.sh dist -t aarch64
  ./build.sh install
  ./build.sh install /opt/unitree
  ./build.sh all
EOF
}

task_sdk() { "$ROOT/scripts/build_sdk.sh" "$@"; }
task_dist() { "$ROOT/scripts/make_dist.sh" "$@"; }
task_install() { "$ROOT/scripts/install.sh" "$@"; }
task_clean() { "$ROOT/scripts/clean.sh"; }

task_all() {
    local arch
    for arch in x86_64 aarch64; do
        "$ROOT/scripts/build_sdk.sh" -t "$arch" "$@"
        "$ROOT/scripts/make_dist.sh" -t "$arch"
    done
}

main() {
    local task=${1:-}
    if [ $# -gt 0 ]; then
        shift
    fi
    case "$task" in
        sdk)     task_sdk "$@" ;;
        dist)    task_dist "$@" ;;
        install) task_install "$@";;
        all)     task_all "$@" ;;
        clean)   task_clean ;;
        -h|--help|"") usage ;;
        *) usage >&2; return 2 ;;
    esac
}

main "$@"
