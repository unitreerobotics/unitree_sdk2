#!/usr/bin/env bash
# Remove all generated artifacts from this repository:
# buildout/ (local staging) and dist/ (deliverables).
# Source trees and downloaded toolchain packages are not removed: they are not
# artifacts, and re-downloading them is expensive.
#   ./build.sh clean
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

remove() {  # remove <dir>
  [ -d "$1" ] || { echo "clean: ${1#"$ROOT"/} does not exist, skipping"; return 0; }
  chmod -R u+w "$1"   # dist/ is read-only after assembly; restore write bits first
  rm -rf "$1"
  echo "clean: removed ${1#"$ROOT"/}"
}

remove "$ROOT/buildout"
remove "$ROOT/dist"
