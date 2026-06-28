# nbody-datoviz

A drop-in comparison counterpart to [`samples/nbody`](../nbody): the **same** N-body
simulation, rendered with [datoviz](https://github.com/datoviz/datoviz) instead of mimir,
reporting the **same metrics** (plus one extra column) so the two libraries can be compared
head-to-head.

## Why the results differ: the data path

The physics are byte-identical (same CUDA kernel, CPU integrator, body initialization, RNG
seed and parameters). The only difference is how rendered positions reach the GPU:

| | mimir | datoviz |
|---|---|---|
| Per-frame transfer | **none** — CUDA and Vulkan share one buffer (`allocLinear`) | **GPU → Host → GPU** every frame |
| PCIe crossings / frame | 0 | 2 |

datoviz has no CUDA interop (it is on their v0.4+ roadmap). Its API only accepts host
pointers, so each frame the freshly simulated positions must round-trip off and back onto
the GPU. This sample minimizes that unavoidable cost as much as datoviz allows:

1. a GPU kernel packs `float4` → `vec3` (transfer 3 floats/body, not 4; no host repack),
2. `cudaMemcpy` device → **pinned** host buffer (full-bandwidth D2H),
3. `dvz_marker_position` uploads host → GPU (inside the call).

The cost of steps 1–3 is reported in the extra **`transfer_time`** column — the overhead
mimir eliminates entirely.

## Metrics

The CSV columns match `samples/nbody` exactly, with `transfer_time` appended:

```
mode,windowres,N,target_fps,framerate,compute_time,pipeline_time,graphics_time,
vk_usage,vk_budget,gpu_power,gpu_energy,gpu_time,nvml_free,nvml_reserved,nvml_total,
nvml_used,transfer_time
```

Notes on datoviz-specific differences:
- `mode` is `datoviz` (rendering) or `none` (pure simulation, no rendering).
- `pipeline_time` is always `0`: datoviz's internal render-pass GPU time is not exposed
  through its public API.
- `vk_usage` / `vk_budget` are substituted with **NVML used / total (GiB)**, since datoviz
  owns its Vulkan device and mimir's Vulkan memory-budget figures are not reachable. Compute,
  transfer and framerate rows are directly comparable to mimir.

## Building

datoviz is built from source via CMake `ExternalProject` (pinned to `v0.3.5`; override with
`-DDATOVIZ_TAG=...`). datoviz's own CMake hardcodes `${CMAKE_SOURCE_DIR}` and ships its Vulkan
loader + `glslc` at fixed in-tree paths, so it cannot be consumed with `FetchContent_MakeAvailable`
/ `add_subdirectory`; we build it as a separate project and link the resulting `libdatoviz.so`.
The first build is slow because datoviz clones its submodules (imgui, data) and FetchContent
dependencies (cglm, glfw). We build it with `DATOVIZ_WITH_MSDF=OFF` (skips the flaky msdfgen build;
this sample only uses disc markers) and `DATOVIZ_WITH_CLI=OFF`.

### Prerequisites

A Vulkan SDK is **not** required — datoviz bundles `glslc` and `libvulkan.so`. But glfw is built
from source on Linux and needs the X11 development headers. On Debian/Ubuntu:

```sh
sudo apt install xorg-dev   # libxrandr-dev, libxinerama-dev, libxcursor-dev, libxi-dev, ...
```

(zlib is optional: datoviz only uses it for gzipped test data, which this sample never loads.)

**Compiler**: CUDA 12/13 does not support GCC 16+. On systems where the default `gcc` is newer
than GCC 14 (e.g. Arch Linux), pass the versioned compilers explicitly:

```sh
cmake -B build \
    -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-14
```

This sample is OFF by default in the samples tree; enable it with
`-DMIMIR_SAMPLES_BUILD_DATOVIZ=ON`, or build this directory standalone:

```sh
cmake -B build \
    -DCMAKE_CXX_COMPILER=g++-14 \
    -DCMAKE_C_COMPILER=gcc-14 \
    -DCMAKE_CUDA_HOST_COMPILER=g++-14
cmake --build build -j
```

## Running

Same CLI argument order as `samples/nbody`:

```
benchmark_datoviz <width> <height> <body_count> <iters> <present> <target_fps> <enable_sync> <display> <use_cpu>
```

Or use the batch driver (writes a CSV with the header):

```sh
./batch_main.sh results.csv
```
