#!/bin/bash

set -e

ARCH=$(uname -m)
## Write compilation script for Release - local architecture
OUTFILE=$1/Builds/linux-x86_64/compileRelease.sh
echo -e '#!/bin/bash\nmake -j`nproc` CONFIG=Release' >$OUTFILE
chmod +x $OUTFILE

# Look for the first uncommented line in file $1/elk.version
ELK_VERSION=$(echo $(grep -Po "^[^0-9#]*\d+\.\d+\.\d+" $1/elk.version | head -n 1) | xargs )
# echo Elk-pi version read from the 'elk.version' file: \"$ELK_VERSION\"

TOOLCHAIN=""
# If ELK_VERSION is empty
if [ -z "$ELK_VERSION" ]; then
    TOOLCHAIN="echo "elk.version file not found or empty. Please, check the file and try again.""
else
    # If ELK_VERSION is 0.11.0
    if [ "$ELK_VERSION" == "0.11.0" ]; then
        TOOLCHAIN="# source /opt/elk/1.0/environment-setup-aarch64-elk-linux # Elk-pi v0.9.0\nsource /opt/elk/0.11.0/environment-setup-cortexa72-elk-linux # Elk-pi v0.11.0 \n# source /opt/elk/1.0.0/environment-setup-cortexa72-elk-linux # Elk-pi v1.0.0 "
    else
        # If ELK_VERSION is 1.0.0
        if [ "$ELK_VERSION" == "1.0.0" ]; then
            TOOLCHAIN="# source /opt/elk/1.0/environment-setup-aarch64-elk-linux # Elk-pi v0.9.0\n# source /opt/elk/0.11.0/environment-setup-cortexa72-elk-linux # Elk-pi v0.11.0 \nsource /opt/elk/1.0.0/environment-setup-cortexa72-elk-linux # Elk-pi v1.0.0 "
        else
            if [ "$ELK_VERSION" == "0.9.0" ]; then
                TOOLCHAIN="source /opt/elk/1.0/environment-setup-aarch64-elk-linux # Elk-pi v0.9.0\n# source /opt/elk/0.11.0/environment-setup-cortexa72-elk-linux # Elk-pi v0.11.0 \n# source /opt/elk/1.0.0/environment-setup-cortexa72-elk-linux # Elk-pi v1.0.0 \nsource /opt/elk/1.1.0/environment-setup-cortexa72-elk-linux # Elk-pi v1.1.0 "
            else
                echo "Elk-pi version not supported. Please, check the 'elk.version' file and try again."
                exit 1
            fi
        fi
    fi
fi

INSTRUCTIONS='#!/bin/bash\n# Cross-compilation script for ElkOS on RPI4\n# Instructions at:\n# https://elk-audio.github.io/elk-docs/html/documents/building_plugins_for_elk.html#cross-compiling-juce-plugin\nunset LD_LIBRARY_PATH\n'
RENAME_VST3_ARCH='# Rename VST3 inner path to match the subfolder expexted by elkos\nfor f in `find build -name "*.vst3"`; do rm -rf $f/Contents/aarch64-linux && mv $f/Contents/arm64-linux $f/Contents/aarch64-linux; done'
ADDITIONAL_OPTIMIZATIONS='export CXXFLAGS="-O3 -pipe -ffast-math -feliminate-unused-debug-types -funroll-loops"'

## Write compilation script for Release - aarch64
OUTFILE=$1/Builds/linux-aarch64/compileReleaseElkPi4.sh
echo -e $INSTRUCTIONS'\n'$TOOLCHAIN'\n\n'$ADDITIONAL_OPTIMIZATIONS'
AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Release CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"
\n'$RENAME_VST3_ARCH'' >$OUTFILE
chmod +x $OUTFILE

## Write compilation script for Debug - aarch64
OUTFILE=$1/Builds/linux-aarch64/compileDebugElkPi4.sh
echo -e $INSTRUCTIONS'\n'$TOOLCHAIN'\n\n'$ADDITIONAL_OPTIMIZATIONS'
AR=aarch64-elk-linux-ar make -j`nproc` CONFIG=Debug CFLAGS="-DJUCE_HEADLESS_PLUGIN_CLIENT=1 -Wno-psabi" TARGET_ARCH="-mcpu=cortex-a72 -mtune=cortex-a72"
\n'$RENAME_VST3_ARCH'' >$OUTFILE
chmod +x $OUTFILE
