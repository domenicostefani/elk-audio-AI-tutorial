#!/bin/bash
# Cross-compilation script for ElkOS on RPI4
# Instructions at:
# https://elk-audio.github.io/elk-docs/html/documents/building_plugins_for_elk.html#cross-compiling-juce-plugin

set -e # Exit on error

mkdir -p tensorflow-build-aarch64
cd tensorflow-build-aarch64

# If file ../tensorflow/lite/tools/cmake/modules/flatbuffers.cmake contains 'GIT_TAG v2.0.6' then we auto replace it with 'GIT_TAG v2.0.8'
# This solves a bug in version 2.0.6 of Flatbuffers that causes errors with cross-compilation
# https://github.com/tensorflow/tensorflow/issues/57617s
sed -i 's/GIT_TAG v2.0.6/GIT_TAG v2.0.8 #https:\/\/github.com\/tensorflow\/tensorflow\/issues\/57617/g' ../tensorflow/tensorflow/lite/tools/cmake/modules/flatbuffers.cmake

unset LD_LIBRARY_PATH
source /opt/elk/0.11.0/environment-setup-cortexa72-elk-linux

cmake ../tensorflow/tensorflow/lite -DTFLITE_ENABLE_XNNPACK=OFF -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake

export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops -Wno-poison-system-directories"

AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Release CFLAGS="-Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"
