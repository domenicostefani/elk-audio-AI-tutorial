#!/bin/bash
set -e # Exit on error

AARCH_NAME=$(uname -m)

mkdir -p tensorflow-build-$AARCH_NAME
cd tensorflow-build-$AARCH_NAME

cmake ../tensorflow/tensorflow/lite -DTFLITE_ENABLE_XNNPACK=OFF

export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops -Wno-poison-system-directories"

make -j`nproc` CONFIG=Release