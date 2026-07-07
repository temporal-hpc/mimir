#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
GCC_VERSION=""
BUILD_DIR="$SCRIPT_DIR/build"
BUILD_TYPE="Release"
ENABLE_REMOTE=OFF
ENABLE_QUIC=OFF
HEADLESS=OFF
JOBS=$(nproc)

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Incremental build of the mimir library. Unlike mimir-build-from-zero.sh, this
does NOT delete the build directory: it reuses it and only recompiles what
changed. On a first run (no build dir yet) it configures from scratch.

The configure flags below are only applied when configuring a fresh build dir;
for an existing build dir CMake reuses its cached configuration and these are
ignored. To change flags, use mimir-build-from-zero.sh (or delete build/).

Options:
  --gcc <version>    GCC version to use as the CUDA host compiler (e.g. 14).
                     Required on systems where the default GCC exceeds CUDA's
                     supported version (e.g. Arch Linux with GCC 16).
  --remote           Enable H.264 remote streaming (MIMIR_ENABLE_REMOTE=ON).
                     Requires: ffmpeg (libavcodec, libavutil, libswscale).
  --quic             Enable QUIC transport (MIMIR_ENABLE_QUIC=ON).
                     Requires: ngtcp2 + OpenSSL.
  --headless         Build without X11/display support (for HPC nodes / containers
                     with no display stack). rr-server still works; windowed samples won't.
  --debug            Build in Debug mode (default: Release).
  --build-dir <dir>  Build directory (default: build/).
  --jobs <n>         Parallel build jobs (default: $(nproc)).
  -h, --help         Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --gcc 14
  $(basename "$0") --gcc 14 --jobs 8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gcc)        GCC_VERSION="$2";  shift 2 ;;
        --remote)     ENABLE_REMOTE=ON;  shift ;;
        --quic)       ENABLE_QUIC=ON;    shift ;;
        --headless)   HEADLESS=ON;       shift ;;
        --debug)      BUILD_TYPE=Debug;  shift ;;
        --build-dir)  BUILD_DIR="$2";    shift 2 ;;
        --jobs)       JOBS="$2";         shift 2 ;;
        -h|--help)    usage; exit 0 ;;
        *)
            echo "Error: unknown option: $1" >&2
            echo ""
            usage
            exit 1 ;;
    esac
done

CMAKE_ARGS=(
    -S "$SCRIPT_DIR"
    -B "$BUILD_DIR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DMIMIR_ENABLE_REMOTE="$ENABLE_REMOTE"
    -DMIMIR_ENABLE_QUIC="$ENABLE_QUIC"
    -DMIMIR_HEADLESS="$HEADLESS"
)

if [[ -n "$GCC_VERSION" ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_C_COMPILER="/usr/bin/gcc-${GCC_VERSION}"
        -DCMAKE_CXX_COMPILER="/usr/bin/g++-${GCC_VERSION}"
        -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++-${GCC_VERSION}"
    )
fi

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "==> No existing build in $BUILD_DIR; configuring from scratch..."
    cmake "${CMAKE_ARGS[@]}"
else
    echo "==> Reusing existing build directory: $BUILD_DIR"
fi

echo "==> Building with $JOBS jobs..."
cmake --build "$BUILD_DIR" -j "$JOBS"

echo ""
echo "Done. Library built at: $BUILD_DIR"
echo "MIMIR_DIR for samples:  $BUILD_DIR/lib/mimir"
echo ""
echo "To build a sample, run:"
echo "  ./samples-build-from-change.sh --sample <name> [--gcc $GCC_VERSION]"
