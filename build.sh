#!/bin/sh
# Builds the C++ core and whatever currently exercises it, into build/.
#
#     ./build.sh              the parity checks
#     ./build.sh --debug      unoptimized, with assertions and sanitizers
#
# No make and no cmake: the core is a handful of files and one g++ line each.
#
# -static is not a preference.  Linking libstdc++ dynamically fails on this
# machine's MSYS2 with a bare "ld returned 116", while a static link works and
# owes no DLLs to whatever runs it.
#
# No -ffast-math, ever.  It let LLVM reorder a store ahead of a load in the
# numba kernels and every bucket maximum read back zero; see kernels.py.

set -e
cd "$(dirname "$0")"
mkdir -p build

FLAGS="-std=c++17 -Wall -Wextra -static"
if [ "$1" = "--debug" ]; then
    FLAGS="$FLAGS -O0 -g -fsanitize=undefined"
else
    FLAGS="$FLAGS -O2"
fi

CORE="core/words.cpp"

echo "building with: $FLAGS"
g++ $FLAGS -o build/parity_feedback.exe native/parity_feedback.cpp $CORE
echo "built build/parity_feedback.exe"
