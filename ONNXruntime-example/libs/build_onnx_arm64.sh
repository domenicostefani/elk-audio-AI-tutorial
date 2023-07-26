#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

unset LD_LIBRARY_PATH
source /opt/elk/0.11.0/environment-setup-cortexa72-elk-linux

# CONFIGURATION=Release
CONFIGURATION=MinSizeRel

PROTOBUF_v="3.11.3"
PROTOBUF="protoc-"$PROTOBUF_v"-linux-aarch_64"

#Check if folder $PROTOBUF exists, otherwise download and unzip
if [ ! -d "$PROTOBUF" ]; then
    wget https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF_v/$PROTOBUF.zip
    unzip $PROTOBUF.zip -d $PROTOBUF
    export PATH=$PATH:$PWD/$PROTOBUF/bin
    export PATH=$PATH:$PWD/$PROTOBUF/include
    rm $PROTOBUF.zip
fi

PATH_TO_PROTOC=$PWD/$PROTOBUF/bin/protoc

echo -n "" > cmake.tool
echo "SET(CMAKE_SYSTEM_NAME Linux)" >>cmake.tool
echo "SET(CMAKE_SYSTEM_VERSION 1)" >>cmake.tool
echo "SET(CMAKE_C_COMPILER aarch64-elk-linux-gcc)" >>cmake.tool
echo "SET(CMAKE_CXX_COMPILER aarch64-elk-linux-g++)" >>cmake.tool
echo "SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)" >>cmake.tool
echo "SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)" >>cmake.tool
echo "SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)" >>cmake.tool
echo "SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)" >>cmake.tool
echo "SET(CMAKE_FIND_ROOT_PATH /opt/elk/0.11.0/sysroots/cortexa72-elk-linux/)" >>cmake.tool

CMAKETOOLPATH=$(realpath cmake.tool)

mkdir -p onnxruntime/build
cd onnxruntime/build
cmake ../cmake -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON\
               -DCMAKE_BUILD_TYPE=Release \
               -Dprotobuf_WITH_ZLIB=OFF\
               -Donnxruntime_ENABLE_PYTHON=OFF\
               -Donnxruntime_BUILD_SHARED_LIB=OFF\
               -Donnxruntime_DEV_MODE=OFF\
               -DONNX_CUSTOM_PROTOC_EXECUTABLE=$PATH_TO_PROTOC\
               -DCMAKE_TOOLCHAIN_FILE=$CMAKETOOLPATH

            #    -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_TOOLCHAIN_FILE=path/to/tool.cmake -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/mnt/pi/usr/bin/python3 -Donnxruntime_BUILD_SHARED_LIB=OFF -Donnxruntime_DEV_MODE=OFF -DONNX_CUSTOM_PROTOC_EXECUTABLE=/path/to/protoc "-DPYTHON_INCLUDE_DIR=/mnt/pi/usr/include;/mnt/pi/usr/include/python3.7m" -DNUMPY_INCLUDE_DIR=/mnt/pi/folder/to/numpy/headers
make

# AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Release CFLAGS="-Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"
# -Donnxruntime_GCC_STATIC_CPP_RUNTIME=ON -DCMAKE_BUILD_TYPE=Release -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_TOOLCHAIN_FILE=path/to/tool.cmake -Donnxruntime_ENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/mnt/pi/usr/bin/python3 -Donnxruntime_BUILD_SHARED_LIB=OFF -Donnxruntime_DEV_MODE=OFF -DONNX_CUSTOM_PROTOC_EXECUTABLE=/path/to/protoc "-DPYTHON_INCLUDE_DIR=/mnt/pi/usr/include;/mnt/pi/usr/include/python3.7m" -DNUMPY_INCLUDE_DIR=/mnt/pi/folder/to/numpy/headers


# cd onnxruntime; ./build.sh --config $CONFIGURATION --parallel --update --arm64 --build_shared_lib #--target cortex-a72

# cd build/Linux/$CONFIGURATION

# # export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops -Wno-poison-system-directories"
# # AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Release CFLAGS="-Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"

# make -j12