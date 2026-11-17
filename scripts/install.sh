#!/usr/bin/env bash
# Install dist/* point-to-point into a local prefix (default ~/.local/unitree).
#   ./build.sh install [<destination>]
#
# Each dist/<name> is mirrored only to <destination>/<name>; other entries in
# the destination are left untouched.
set -euo pipefail

readonly ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

die() {
    local code=$1
    shift
    printf '%s\n' "$*" >&2
    exit "$code"
}

install_dist() {
    local src=$1 dest=$2
    if [ -d "$dest" ]; then
        chmod -R u+w "$dest"
    fi
    rsync -av --delete-after "$src/" "$dest/"
}

main() {
    local dest=${1:-$HOME/.local/unitree}
    local src name
    case "$dest" in
        ""|"/") die 2 "install: invalid destination '$dest'" ;;
    esac
    command -v rsync >/dev/null 2>&1 || die 3 "install: rsync not found"
    [ -d "$ROOT/dist" ] || die 3 "install: dist/ does not exist (run ./build.sh dist first)"
    mkdir -p "$dest"
    for src in "$ROOT"/dist/*/; do
        [ -d "$src" ] || continue
        name=$(basename "$src")
        install_dist "$src" "$dest/$name"
    done
    printf 'Installed to %s\n' "$dest"
}

main "$@"
