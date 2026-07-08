#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults
GCC_VERSION=""
BUILD_DIR="$SCRIPT_DIR/build"
BUILD_TYPE="Release"
ENABLE_REMOTE=ON
ENABLE_QUIC=ON
HEADLESS=OFF
JOBS=$(nproc)

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Builds the mimir library from scratch, deleting any prior build directory.
All options are optional; run with no arguments to build with defaults.

Options:
  --gcc <version>    GCC version to use as the CUDA host compiler (e.g. 14).
                     Required on systems where the default GCC exceeds CUDA's
                     supported version (e.g. Arch Linux with GCC 16).
  --no-remote        Disable H.264 remote streaming (on by default; needs ffmpeg:
                     libavcodec, libavutil, libswscale — degrades to raw frames if missing).
  --no-quic          Disable the QUIC transport (on by default; needs ngtcp2 + OpenSSL —
                     degrades to TCP-only if missing). Transport/codec are runtime choices.
  --headless         Build without X11/display support (for HPC nodes / containers
                     with no display stack). rr-server still works; windowed samples won't.
  --debug            Build in Debug mode (default: Release).
  --build-dir <dir>  Build directory (default: build/).
  --jobs <n>         Parallel build jobs (default: $(nproc)).
  -h, --help         Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --gcc 14
  $(basename "$0") --gcc 14 --build-dir mybuild --jobs 8
  $(basename "$0") --headless                  # HPC container, no display
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gcc)        GCC_VERSION="$2";  shift 2 ;;
        --remote)     ENABLE_REMOTE=ON;  shift ;; # accepted for compatibility (now the default)
        --quic)       ENABLE_QUIC=ON;    shift ;; # accepted for compatibility (now the default)
        --no-remote)  ENABLE_REMOTE=OFF; shift ;;
        --no-quic)    ENABLE_QUIC=OFF;   shift ;;
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

echo "==> Removing prior build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"

echo "==> Configuring..."
cmake "${CMAKE_ARGS[@]}"

echo "==> Building with $JOBS jobs..."
cmake --build "$BUILD_DIR" -j "$JOBS"

echo ""
echo "Done. Library built at: $BUILD_DIR"
echo "MIMIR_DIR for samples:  $BUILD_DIR/lib/mimir"
echo ""
echo "To build a sample, run:"
echo "  ./samples-build-from-zero.sh --sample <name> [--gcc $GCC_VERSION]"
