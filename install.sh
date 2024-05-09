#!/bin/bash

INSTALL_DIR=${1:-/usr/local}
echo "Installing into $INSTALL_DIR..."

WorkDir=$(cd $(dirname $0); pwd)
echo "WrokDir=$WorkDir"

Arch=$(uname -m)
echo "CPU Arch=$Arch"

ThirdParty=$WorkDir/thirdparty

set -e

mkdir -p "$INSTALL_DIR/include"
mkdir -p "$INSTALL_DIR/lib"

cp -r $WorkDir/include/* "$INSTALL_DIR/include"
cp -r $WorkDir/lib/$Arch/* "$INSTALL_DIR/lib"

cp -r $ThirdParty/include/* "$INSTALL_DIR/include"
cp -r $ThirdParty/lib/$Arch/* "$INSTALL_DIR/lib"

if [[ $INSTALL_DIR == "/usr/local" ]]; then
    ldconfig
fi
