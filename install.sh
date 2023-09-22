#!/bin/bash

WorkDir=$(cd $(dirname $0); pwd)
echo "WrokDir=$WorkDir"

Arch=$(uname -m)
echo "CPU Arch=$Arch"

ThirdParty=$WorkDir/thirdparty

set -e

cp -r $WorkDir/include/* /usr/local/include
cp -r $WorkDir/lib/$Arch/* /usr/local/lib

cp -r $ThirdParty/include/* /usr/local/include
cp -r $ThirdParty/lib/$Arch/* /usr/local/lib

ldconfig
