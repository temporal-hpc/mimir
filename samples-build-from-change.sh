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
    particles-kmodal-3d
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
STAGE_SHADERS=OFF

usage() {
    local samples_list
    samples_list=$(printf '    %s\n' "${AVAILABLE_SAMPLES[@]}")

    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Incremental build of mimir samples. Unlike samples-build-from-zero.sh, this does
NOT delete the build directory: it reuses it and only recompiles what changed.
On a first run (no build dir yet) it configures from scratch.
The mimir library must already be built (run mimir-build-from-change.sh first).

The configure flags below are only applied when configuring a fresh build dir;
for an existing build dir CMake reuses its cached configuration and these are
ignored. To change flags, use samples-build-from-zero.sh (or delete the build dir).

Options:
  --sample <name>    Build only this sample (default: build all).
                     Available samples:
$samples_list
  --mimir-dir <path> Path to the folder containing mimirConfig.cmake.
                     Default: build/lib/mimir/ (relative to repo root).
                     Set this if you installed mimir elsewhere, e.g.:
                       /usr/local/lib/cmake/mimir
  --gcc <version>    GCC version to use as CUDA host compiler (e.g. 14).
                     Pass the same version you used to build the library.
  --build-dir <dir>  Build directory.
                     Default (all samples):   samples/build/
                     Default (single sample): samples/<name>/build/
  --jobs <n>         Parallel build jobs (default: $(nproc)).
  --stage-shaders    After building, force-refresh the staged shaders/ dir next
                     to each built binary from the repo shaders/. A shader-only
                     edit does not relink the target, so CMake's POST_BUILD copy
                     never fires and the binary keeps running the stale staged
                     shader -- pass this to iterate on .slang files without a relink.
  -h, --help         Show this help.

Examples:
  $(basename "$0")
  $(basename "$0") --gcc 14
  $(basename "$0") --sample remote-rendering --gcc 14
  $(basename "$0") --sample particles-kmodal-3d --stage-shaders  # after a .slang edit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)    SAMPLE="$2";    shift 2 ;;
        --mimir-dir) MIMIR_DIR="$2"; shift 2 ;;
        --gcc)       GCC_VERSION="$2"; shift 2 ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --jobs)      JOBS="$2";      shift 2 ;;
        --stage-shaders) STAGE_SHADERS=ON; shift ;;
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
    echo "    ./mimir-build-from-change.sh [--gcc <version>]" >&2
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

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    echo "==> No existing build in $BUILD_DIR; configuring from scratch..."
    cmake "${CMAKE_ARGS[@]}"
else
    echo "==> Reusing existing build directory: $BUILD_DIR"
fi

# ── Force a relink when the mimir static library is newer than a built sample ──────────────────
# The mimir library links into samples as a static archive (libmimir.a). After you rebuild the
# library (e.g. mimir-build-from-change.sh following an edit under lib/), the sample binary here is
# stale until it relinks. CMake's imported-target dependency on the .a does not always trigger that
# relink, which silently leaves the sample running against the old library -- the exact "I forgot to
# recompile" trap. Guard it: if the .a is newer than an already-built ELF binary in this build dir,
# delete that binary so the build below relinks it. The object files are untouched, so this is a
# link step only (fast), not a recompile. Skipped on a fresh build dir (nothing to be stale yet).
LIB_A="$(cd "$MIMIR_DIR/.." 2>/dev/null && pwd)/libmimir.a"
if [[ -f "$LIB_A" && -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    while IFS= read -r bin; do
        # only real ELF binaries, and only when the library is strictly newer than the binary
        if [[ "$LIB_A" -nt "$bin" ]] && head -c4 "$bin" 2>/dev/null | grep -qa 'ELF'; then
            echo "==> libmimir.a is newer than $(basename "$bin"); removing it to force a relink"
            rm -f "$bin"
        fi
    done < <(find "$BUILD_DIR" -maxdepth 2 -type f -perm -u+x 2>/dev/null)
fi

echo "==> Building with $JOBS jobs..."
cmake --build "$BUILD_DIR" -j "$JOBS"

# ── Stage shaders (shader-only edits don't relink -> POST_BUILD copy never fires) ─
if [[ "$STAGE_SHADERS" == "ON" ]]; then
    REPO_SHADERS="$SCRIPT_DIR/shaders"
    if [[ ! -d "$REPO_SHADERS" ]]; then
        echo "Error: repo shaders dir not found: $REPO_SHADERS" >&2
        exit 1
    fi
    # Refresh every staged shaders/ dir already sitting next to a built binary
    # (CMake stages to <binary-dir>/shaders, which is build/bin or build depending
    # on the sample). Skip nested */shaders/* so we only touch the top-level dirs.
    mapfile -t SHADER_DESTS < <(find "$BUILD_DIR" -type d -name shaders -not -path "*/shaders/*")
    if [[ ${#SHADER_DESTS[@]} -eq 0 ]]; then
        echo "==> --stage-shaders: no staged shaders/ dir found under $BUILD_DIR." >&2
        echo "    Run a full build first so each binary + its shaders/ dir exist." >&2
    else
        for dest in "${SHADER_DESTS[@]}"; do
            echo "==> Staging shaders -> $dest"
            cmake -E copy_directory "$REPO_SHADERS" "$dest"
        done
    fi
fi

echo ""
if [[ -n "$SAMPLE" ]]; then
    echo "Done. Binary/binaries in: $BUILD_DIR/"
else
    echo "Done. Binaries in: $BUILD_DIR/bin/"
fi
