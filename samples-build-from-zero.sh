#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SAMPLES_DIR="$SCRIPT_DIR/samples"

AVAILABLE_SAMPLES=(
    CA3D-voxels
    codigo_nv
    colloids
    edgeflip
    GPU-visual-tool-example
    headless
    image
    mesh3d
    nbody
    points3d
    potts
    powermon
    remote-rendering
    structured
    texture_cudaarray
    unstructured
    voronoi2
    voronoi-simple
)

# Defaults
SAMPLE=""
MIMIR_DIR="$SCRIPT_DIR/build/lib/mimir"
GCC_VERSION=""
BUILD_DIR=""
JOBS=$(nproc)

usage() {
    local samples_list
    samples_list=$(printf '    %s\n' "${AVAILABLE_SAMPLES[@]}")

    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Builds mimir samples, either all at once or a single one.
All options are optional; run with no arguments to build all samples with defaults.
The mimir library must already be built (run mimir-build-from-zero.sh first).

Options:
  --sample <name>    Build only this sample (default: build all).
                     Available samples:
$samples_list
  --mimir-dir <path> Path to the folder containing mimirConfig.cmake.
                     Default: build/lib/mimir/ (relative to repo root).
                     Set this if you installed mimir elsewhere, e.g.:
                       /usr/local/lib/cmake/mimir
  --gcc <version>    GCC version to use as CUDA host compiler (e.g. 14).
                     Pass the same version you used in mimir-build-from-zero.sh.
  --build-dir <dir>  Build directory.
                     Default (all samples):   samples/build/
                     Default (single sample): samples/<name>/build/
  --jobs <n>         Parallel build jobs (default: $(nproc)).
  -h, --help         Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --gcc 14
  $(basename "$0") --sample remote-rendering --gcc 14
  $(basename "$0") --sample headless --mimir-dir /usr/local/lib/cmake/mimir
  $(basename "$0") --mimir-dir \$(pwd)/build/lib/mimir --jobs 8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)    SAMPLE="$2";    shift 2 ;;
        --mimir-dir) MIMIR_DIR="$2"; shift 2 ;;
        --gcc)       GCC_VERSION="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --jobs)      JOBS="$2";      shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)
            echo "Error: unknown option: $1" >&2
            echo ""
            usage
            exit 1 ;;
    esac
done

# ── Validate mimir-dir ────────────────────────────────────────────────────────
if [[ ! -f "$MIMIR_DIR/mimirConfig.cmake" ]]; then
    echo "Error: mimirConfig.cmake not found at: $MIMIR_DIR" >&2
    echo "" >&2
    echo "  Make sure you have built the mimir library first:" >&2
    echo "    ./mimir-build-from-zero.sh [--gcc <version>]" >&2
    echo "" >&2
    echo "  Then point --mimir-dir at the folder containing mimirConfig.cmake." >&2
    echo "  After a local build that is: build/lib/mimir/" >&2
    exit 1
fi

# ── Validate --sample if given ────────────────────────────────────────────────
if [[ -n "$SAMPLE" ]]; then
    if [[ ! -d "$SAMPLES_DIR/$SAMPLE" ]]; then
        echo "Error: unknown sample: '$SAMPLE'" >&2
        echo "" >&2
        echo "Available samples:" >&2
        printf '  %s\n' "${AVAILABLE_SAMPLES[@]}" >&2
        exit 1
    fi
fi

# ── Resolve build directory ───────────────────────────────────────────────────
if [[ -z "$BUILD_DIR" ]]; then
    if [[ -n "$SAMPLE" ]]; then
        BUILD_DIR="$SAMPLES_DIR/$SAMPLE/build"
    else
        BUILD_DIR="$SAMPLES_DIR/build"
    fi
fi

# ── Resolve source directory ──────────────────────────────────────────────────
if [[ -n "$SAMPLE" ]]; then
    SOURCE_DIR="$SAMPLES_DIR/$SAMPLE"
else
    SOURCE_DIR="$SAMPLES_DIR"
fi

# ── Build CMAKE_ARGS ──────────────────────────────────────────────────────────
CMAKE_ARGS=(
    -S "$SOURCE_DIR"
    -B "$BUILD_DIR"
    -Dmimir_DIR="$MIMIR_DIR"
)

if [[ -n "$GCC_VERSION" ]]; then
    CMAKE_ARGS+=(
        -DCMAKE_C_COMPILER="/usr/bin/gcc-${GCC_VERSION}"
        -DCMAKE_CXX_COMPILER="/usr/bin/g++-${GCC_VERSION}"
        -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++-${GCC_VERSION}"
    )
fi

# ── Print what we're about to do ─────────────────────────────────────────────
if [[ -n "$SAMPLE" ]]; then
    echo "==> Building sample: $SAMPLE"
else
    echo "==> Building all samples"
fi
echo "    mimir-dir : $MIMIR_DIR"
echo "    build-dir : $BUILD_DIR"
[[ -n "$GCC_VERSION" ]] && echo "    gcc       : $GCC_VERSION"
echo ""

echo "==> Removing prior build directory: $BUILD_DIR"
rm -rf "$BUILD_DIR"

echo "==> Configuring..."
cmake "${CMAKE_ARGS[@]}"

echo "==> Building with $JOBS jobs..."
cmake --build "$BUILD_DIR" -j "$JOBS"

echo ""
if [[ -n "$SAMPLE" ]]; then
    echo "Done. Binary/binaries in: $BUILD_DIR/"
else
    echo "Done. Binaries in: $BUILD_DIR/bin/"
fi
