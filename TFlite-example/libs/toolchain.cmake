# targets
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
# Setting CMAKE_SYSTEM_PROCESSOR here solves a bug with Abseil 
# not accepting the value cortexa72 that gets set by (probably) Elk's toolchain file.
# aarch64 is correct for ARM64 and is accepted by Abseil.

# Somehow related https://github.com/abseil/abseil-cpp/issues/365

# A separate toolchain was not needed at all when TFLite was just a 
# dependency of a different project, and in that CMakeLists.txt file CMAKE_SYSTEM_PROCESSOR was set to aarch64.
# Apparently in that situation this superseeds all values of CMAKE_SYSTEM_PROCESSOR for all dependencies.
#
# However, when TFLite is compiled by itself, setting CMAKE_SYSTEM_PROCESSOR does not seem enough.
# Abseil still showed errors.
#
# Many thanks to @rodrigodzf