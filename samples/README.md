# Mìmir samples

Each sample is a self-contained CMake project that links against the mimir library.
You can build all samples at once using the umbrella project in this folder, or build
any single sample on its own.

## Prerequisites

Build the mimir library first. From the repository root:

```sh
./mimir-build-from-zero.sh
```

**Or, manually:**
```sh
cmake -B build
cmake --build build -j
```

> **GCC compatibility:** CUDA supports GCC up to a maximum version per release. Check yours with
> `nvcc --version` and use `--gcc <N>` (script) or `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-<N>`
> (cmake) if your default GCC exceeds it. For the authoritative table see the
> [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/).
>
> | CUDA | Max GCC |
> |---|---|
> | 12.0 – 12.2 | 12 |
> | 12.3 – 12.6 | 13 |
> | 13.x | 14 |
>
> _Arch Linux as of June 2026 defaults to GCC 16 with CUDA 13.x — use `--gcc 14`._

## Building all samples

From the repository root, after building the library:

```sh
./samples-build-from-zero.sh   # add --gcc 14 if your system needs it
```

**Or, manually:**
```sh
cmake -B samples/build -S samples/ -Dmimir_DIR=$(pwd)/build/lib/mimir
cmake --build samples/build -j
```

Binaries land in `samples/build/bin/`. Shaders are copied there automatically.

## Building a single sample

```sh
./samples-build-from-zero.sh --sample points3d   # add --gcc 14 if your system needs it
```

**Or, manually:**
```sh
cmake -B samples/points3d/build -S samples/points3d/ -Dmimir_DIR=$(pwd)/build/lib/mimir
cmake --build samples/points3d/build -j
```

The binary lands in `samples/points3d/build/`. Run it from there — shaders are already next to it.

Run `./samples-build-from-zero.sh --help` to see all available sample names and options.

## Running samples (umbrella build)

Run from `samples/build/bin/` — shaders are already there:

```sh
cd samples/build/bin
./run_points3d
```

## Pointing samples at a custom library location

The scripts default to using the library from `build/lib/mimir/` (the local build). If you
installed mimir elsewhere, pass `--mimir-dir`:

```sh
./samples-build-from-zero.sh --mimir-dir ~/.local/lib/cmake/mimir
```

`MIMIR_DIR` must always point to the folder containing `mimirConfig.cmake`:

```
build/
  lib/
    mimir/
      mimirConfig.cmake   ← this folder
```

You can also set it as an environment variable or pass `-Dmimir_DIR=<path>` directly to cmake
for manual builds.

## Available samples

| Binary | Description |
|---|---|
| `run_unstructured` | 2D brownian moving point cloud with various point sizes |
| `run_structured` | Point cloud + Jump Flood Algorithm CUDA kernel for Distance Transform |
| `run_image [path]` | Image viewer and box filter with periodically varying radii |
| `run_mesh [path.obj]` | Triangle mesh loader; CUDA kernel deforms the mesh along vertex normals |
| `run_automata3d` | 3D cellular automaton (CA3D voxels) |
| `run_voronoi` | Simple Voronoi diagram |
| `voronoi` | Voronoi diagram (advanced) |
| `run_sync` | GPU visual tool example (synchronous rendering) |
| `tiuque` | Edge-flip mesh processing (edgeflip) |
| `run_texture` | CUDA array texture viewer (ported from vulkanImageCUDA) |
| `benchmark` | Gravitational N-body simulation (nbody) |
| `run_points3d` | 3D point cloud with power monitoring |
| `potts3` | Potts model visualization |
| `run_colloids [params]` | 2D colloidal particle system with excluded volumes |
| `rr-server` / `rr-client` | Remote rendering — see below |

### Colloids parameters

Recommended parameter sets for `run_colloids`:
```
4096 0.79 0.01 0.7 1 1 0.3 1 -1
4096 0.52 0.01 0.5 1 1 0.5 -1 -2
4096 0.0873 0.01 0.5 1 1 0.5 -1 -4
```

### Remote rendering sample

`rr-server` and `rr-client` require extra system packages and the library must be built
with additional feature flags — the plain `cmake -B build` from the Prerequisites section
above is not enough.

**Extra dependencies:** ffmpeg (H.264) and ngtcp2 + OpenSSL (QUIC).
On Arch: `pacman -S ffmpeg libngtcp2 openssl`.

**Step 1 — rebuild the library** with the remote rendering flags:
```sh
./mimir-build-from-zero.sh --remote --quic   # add --gcc 14 if needed
```
**Or, manually:**
```sh
cmake -B build -DMIMIR_ENABLE_REMOTE=ON -DMIMIR_ENABLE_QUIC=ON
cmake --build build -j
```

**Step 2 — build the sample:**
```sh
./samples-build-from-zero.sh --sample remote-rendering   # add --gcc 14 if used in step 1
```
**Or, manually:**
```sh
cmake -B samples/remote-rendering/build -S samples/remote-rendering/ -Dmimir_DIR=$(pwd)/build/lib/mimir
cmake --build samples/remote-rendering/build -j
```

**Step 3 — run** (server on the GPU box, client on the viewing machine):
```sh
cd samples/remote-rendering/build
./rr-server 9000 1280 720 50000 1 quic
./rr-client <server-ip> 9000
```

Full deployment details (auth token, SSH tunnel, controls) are in
[`samples/remote-rendering/README.md`](remote-rendering/README.md).
