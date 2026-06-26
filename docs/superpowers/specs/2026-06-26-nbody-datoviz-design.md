# nbody-datoviz — Design

**Date:** 2026-06-26
**Status:** Approved (design); pending implementation plan

## 1. Goal & scope

Add a new sample `samples/nbody-datoviz/` that runs the **identical N-body simulation**
as `samples/nbody/` but renders with **datoviz** (https://github.com/datoviz/datoviz)
instead of mimir. It reports the **same metrics** as `nbody` (plus one extra column) so
the two libraries can be compared head-to-head. The only intended difference is the
rendering library and the data path it forces.

### Key constraint discovered during design

Datoviz has **no CUDA–Vulkan interop** — it is on their v0.4+ roadmap, not implemented.
Its public API only accepts **host pointers** for visual data
(`dvz_marker_position(visual, 0, n, host_ptr, 0)`). Therefore datoviz cannot render
directly from a CUDA device buffer the way mimir does with `allocLinear` (shared
external GPU memory, zero transfer). Each frame the simulation result must make a full
**GPU → Host → GPU** round trip:

1. **GPU → Host** — our explicit `cudaMemcpy` D2H from the CUDA position buffer into host RAM.
2. **Host → GPU** — datoviz's `dvz_marker_position` uploads that host array back into its
   Vulkan vertex buffer (hidden inside the call, but it happens).

This round trip is the headline cost the comparison demonstrates: mimir = 0 PCIe
crossings/frame, datoviz = 2.

## 2. Simulation parity (copied verbatim from nbody)

So the physics are byte-identical:

- `nbody_gpu.cu` / `nbody_gpu.cuh` — same `integrateBodies` kernel, block size 256.
- `nbody_cpu.cpp` / `nbody_cpu.hpp` — same CPU integrator (for the `use_cpu` path).
- `randomizeBodies`, `demo_params[3]`, `NBodyParams`, softening — same init, same RNG
  seed (12345), same parameters.
- Same CLI argument order:
  `width height body_count iter_count present target_fps enable_sync display use_cpu`,
  so `batch_main.sh` works by changing only the binary path.
- powermon (NVML) integration identical.

## 3. Data path (the crux)

- CUDA buffers are **plain `cudaMalloc`** ping-pong `dPos[0]` / `dPos[1]` plus `dVel`
  (no `allocLinear`, no interop).
- A **pinned host buffer** (`cudaHostAlloc`, `float4 × body_count`) stages the D2H copy
  at full PCIe bandwidth.
- Per iteration:
  1. CUDA event start → `integrateNbodySystem` kernel writes `dPos[write]` → CUDA event stop (compute time).
  2. `std::swap(read, write)`.
  3. Transfer timer start → `cudaMemcpy` D2H `dPos[read]` → pinned host →
     `dvz_marker_position(visual, 0, n, host, 0)` (datoviz uploads H2D) → transfer timer stop.
  4. Graphics timer start → `dvz_app_step(app)` (render + present one frame) → graphics timer stop.

## 4. Datoviz rendering setup

- `dvz_app(flags)` — `flags = DVZ_APP_FLAGS_OFFSCREEN` when `display == false` (matches
  nbody's headless path); windowed at the requested resolution otherwise.
- `dvz_app_batch` → `dvz_scene` → `dvz_figure(scene, w, h, 0)` → `dvz_panel_default`.
- Visual: **`dvz_marker`** with `dvz_marker_alloc(n)`, `dvz_marker_aspect` / `dvz_marker_shape`
  (round, matching mimir's Markers view), initial `dvz_marker_size` / `dvz_marker_color`.
- 3D camera via `dvz_panel_arcball` (fallback `dvz_panel_panzoom`), positioned to mirror
  nbody's `setCameraPosition({params.x, params.y, params.z})`.
- **Loop control: `dvz_app_step()`** inside our own `for (i < iter_count)` loop — one
  non-blocking frame per simulation iteration, paralleling mimir's manual `updateViews`
  loop (rather than handing control to `dvz_scene_run`).

## 5. Metrics mapping → CSV

Same columns as nbody, **plus a final `transfer_time`** column.

| CSV column | datoviz source |
|---|---|
| `mode` | `"datoviz"` (display) / `"none"` (headless) — parallels mimir's `"mimir"` |
| `resolution`, `N`, `target_fps` | same as nbody |
| `framerate` | host frame-time average over the loop (optionally cross-checked via `dvz_app_timestamps`) |
| `compute_time` | CUDA events around the kernel (same instrument as mimir) |
| `pipeline_time` | **0 / N-A** — datoviz's internal render-pass GPU time is not exposed |
| `graphics_time` | host time of `dvz_app_step` (render + present) |
| `vk_usage` | **NVML used** (GiB) — substitute for Vulkan budget |
| `vk_budget` | **NVML total** (GiB) — substitute |
| `gpu_power`, `gpu_energy`, `gpu_time` | powermon NVML (same as nbody) |
| `nvml_free` / `nvml_reserved` / `nvml_total` / `nvml_used` | NVML (same as nbody) |
| **`transfer_time`** *(new)* | per-frame D2H copy + `dvz_marker_position` upload time, accumulated |

`batch_main.sh` gets the new `transfer_time` header column appended and points at the
datoviz binary.

### Accepted imperfections

- `vk_usage` / `vk_budget` are substituted with NVML used/total in GiB — a slightly
  different quantity than mimir's Vulkan budget, so cross-library *memory* rows are not
  perfectly apples-to-apples. Compute / transfer / framerate rows are directly comparable.
  (User accepted this over leaving them at 0.)
- `pipeline_time` cannot be measured from datoviz's public API; reported as 0.

## 6. Build (FetchContent)

- `samples/nbody-datoviz/CMakeLists.txt` uses
  `FetchContent_Declare(datoviz GIT_REPOSITORY https://github.com/datoviz/datoviz …)`
  pinned to a specific release tag (for reproducibility; latest stable unless specified),
  links `benchmark` against `datoviz`, `powermon`, and CUDA.
- No mimir dependency.
- Added to the umbrella `samples/CMakeLists.txt` so it builds with the rest.
- First build is slow (datoviz pulls its own Vulkan/build deps) — expected.

## 7. To verify during implementation

Exact datoviz signatures against the pinned tag:

- marker aspect/shape enums and round-marker configuration,
- arcball / camera setup and how to set an initial 3D view,
- offscreen `dvz_app_step` semantics,
- whether `dvz_marker_position` triggers an immediate upload or needs a per-step batch
  flush.

These can change the loop details but not the architecture.

## Open items carried into planning

- Pin the exact datoviz release tag.
