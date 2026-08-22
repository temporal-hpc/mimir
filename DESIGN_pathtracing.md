# Design: Vulkan Path Tracing for the particles-kmodal-3d Scene

Status: **Accepted — implementation started** (see HANDOFF.md for exact progress)
Author: Cristobal (drafted with Claude)
Last updated: 2026-07-02

## 1. Goal

Path-trace the particles-kmodal-3d "cheese" scene (k-modal gaussian point clusters in the [-1,1]³
cube, mean-reverting walk) with Vulkan ray tracing, accelerated by RT cores when
available. particles-kmodal-3d completes the benchmark matrix as the render-bound workload:

| sample | compute | transfer | render |
|--------------------|---------|----------|--------|
| nbody | high | low | low |
| cellular automata | low | high | low |
| particles-kmodal-3d (PT) | low | low | **high** |

## 2. Scene and lighting model

- **Sun**: one fixed directional light, world-anchored ("like a sun"). Its initial
  direction points from behind the camera's home position into the scene, and it never
  moves afterwards — in both view modes below.
- **Sky**: the miss shader returns a simple sky gradient (not black), so interiors
  stay readable and the sun is not the only light. Tunable intensity.
- **Materials**: lambertian diffuse, albedo from the per-point color (white today).
- **Post**: exposure + tone map + gamma at the end of the raygen shader.

## 3. View modes (the "rotation effect")

1. **Orbit (default)** — the camera travels around the scene on a turntable orbit;
   simulation cube and sun stay fixed in world space. Auto-rotation with
   `--orbit-speed` (deg/s) so benchmark runs are reproducible without mouse input.
2. **Cube rotation** — the camera and sun stay fixed; the simulation cube's model
   matrix rotates instead. Because the sun is world-anchored, this shows light
   filtering into the cluster geometry from changing angles — different images than
   mode 1, on purpose.

Toggle: CLI `--view-mode 0|1` for scripted runs, plus a key (e.g. `T`) at runtime.

## 4. Geometry: triangle icospheres + instancing (the key decision)

The user requirement is triangle geometry so a real BVH gets built. Naively meshing
1M spheres (80–320 triangles each) means 80–320M triangles and an impossible
per-frame BLAS rebuild. The standard and only practical structure:

- **One unit icosphere BLAS**, built once at startup. `--subdiv` selects tessellation:
  0 = 20 tris, 1 = 80 tris, 2 = 320 tris (default 1).
- **One TLAS with N instances** (`VkAccelerationStructureInstanceKHR`, 64 B each,
  ~64 MB at 1M points): per-instance transform = translate(point) · scale(radius).
- **CUDA writes the instance buffer** directly from the interop position buffer each
  frame (a trivial kernel next to the walk kernel) — the zero-copy story is preserved.
- **TLAS rebuilt (PREFER_FAST_BUILD) every frame** from the updated instance buffer;
  expected a few ms at 1M instances on the RTX 3090 Ti. This build cost is itself a
  headline benchmark number.
- Static scene extras (walls/floor, §8) go in a second, static BLAS + one instance.

## 5. Engine integration (mimir)

New device requirements (currently absent from `getRequiredDeviceExtensions`,
`lib/src/device.cpp:159`): `VK_KHR_acceleration_structure`,
`VK_KHR_ray_tracing_pipeline`, `VK_KHR_deferred_host_operations`,
`VK_KHR_buffer_device_address` (+ the matching feature structs). RT support must be
optional at device pick time so non-RT GPUs still run raster modes.

Render path: raygen writes to a storage image; the existing raster pass blits it and
draws the ImGui HUD on top, so the HUD/metrics machinery is untouched. Shaders are
slang (raygen/miss/closesthit compile to SPIR-V RT stages with the existing
ShaderBuilder), keeping one shader toolchain.

Both workstation GPUs have RT cores (3090 Ti = Ampere gen2, TITAN RTX = Turing gen1).
`VK_KHR_ray_query` fallback for non-RT hardware is out of scope for v1.

## 6. Workload knobs and metrics

CLI: `--spp N` (samples/pixel/frame, default 1), `--bounces N` (max path depth,
default 4), `--subdiv N` (default 1), `--orbit-speed D`, `--view-mode 0|1`.
No progressive accumulation in v1 — the simulation animates every frame, so each
frame is a fresh spp×bounces trace (that IS the benchmark). Accumulate-when-paused
can come later.

New CSV columns (appended, both benchmarks emitting 0 where not applicable):
`tlas_time` (per-frame TLAS build, via VK timestamp queries), `trace_time`
(vkCmdTraceRaysKHR), `spp`, `bounces`, `subdiv`. HUD gains TLAS/Trace ms rows and
a Grays/s estimate.

## 7. Datoviz side

Datoviz has no ray tracing and cannot get it without forking. Recommendation:
benchmark_datoviz stays the **raster baseline** — same simulation, same cloud, same
CSV layout; PT rows come only from benchmark_mimir. The comparison story becomes
"raster reference vs path-traced cost of the identical scene".

## 8. Decisions (confirmed 2026-07-02)

1. **Datoviz role**: raster baseline. Same simulation, same CSV layout; PT rows come
   only from benchmark_mimir.
2. **Scene extras**: interior cube walls (5 diffuse walls, camera side open).
3. **PT home**: mimir engine feature.
4. **Traversal API: RT pipeline + SBT** (not ray query), confirmed 2026-07-02 (later
   session). Both APIs bind the same `VK_KHR_acceleration_structure` hardware and both
   inherit the driver's software-traversal fallback on non-RT GPUs (the "runs on any
   GPU, just slower" behavior — same mechanism that runs DXR on Pascal). Ray query is
   the simpler single-compute-shader route and marginally faster for this one-material
   scene, but RT pipeline + SBT (raygen/miss/closesthit + shader binding table) gives
   per-material shader dispatch, hardware recursion, and callables — the machinery
   future users need for complex multi-material scenes. mimir is built to be prepared
   for that. This also matches the OptiX programming model (raygen/miss/closesthit +
   SBT records) the author already works in. Non-RT-*extension* GPUs still fall back to
   Phong raster; RT support stays optional at device-pick time so raster modes run
   everywhere.
5. **PT workload CLI**: `--spp N` and `--bounces N` as separate, order-independent
   flags (matches the existing `--k` / `--epsilon` style and §6), not positional args
   after `--render-path path-traced`. Ignored unless the render path is path-traced.
6. **Instance writer: Vulkan compute shader, not CUDA** (refines §4), confirmed
   2026-07-03. The per-frame `VkAccelerationStructureInstanceKHR` array is written by an
   engine-owned Vulkan compute shader that reads the interop position buffer in place,
   rather than a CUDA kernel. Rationale: keeps mimir CUDA-kernel-free and the PT feature
   reusable by any sample that supplies an interop position buffer (no per-sample RT/CUDA
   code); a single Vulkan timeline (compute -> AS build -> trace, ordered by barriers in
   one command buffer) instead of an extra CUDA->Vulkan handshake; no duplication of the
   64-byte instance layout in CUDA. The zero-copy interop story is preserved: CUDA still
   writes positions zero-copy, and the compute shader consumes them without a copy. The
   interop timeline wait moves to the COMPUTE stage for PT (the instance writer, not the
   vertex shader, is the first GPU consumer of positions).

### 8.1 Public API: `RenderPath`

The choice is exposed to the programmer as an instance-wide render path — how markers are
turned into pixels, geometry representation and rendering technique together
(`lib/include/public/mimir/options.hpp`):

```cpp
enum class RenderPath { Flat, Impostor, Mesh, PathTraced };
struct ViewerOptions { ...; RenderPath render_path = RenderPath::Impostor; ... };
```

- `Flat`       → unlit raster; markers draw as flat 2D point-sprite discs.
- `Impostor`   → lit raster; markers draw as ray-sphere impostors (historical default).
- `Mesh`       → lit raster; markers draw as instanced triangle icospheres.
- `PathTraced` → the path-traced path of this document.

(This enum was originally `LightModel { None, Phong, PhongMesh, PathTracing }`. The name was
wrong: only `Impostor` vs `PathTraced` is a lighting difference — `Impostor` and `Mesh` share
the same Blinn-Phong shading and differ in geometry. The ordinals are unchanged, so the remote
protocol's render-path field is unaffected.)

`MarkerOptions::render_mode` is engine-managed: `createView` derives it from the instance's
render path (Flat → Flat2D, Impostor → Sphere3D, Mesh → SphereMesh); programs should not set it
directly. On a non-RT device, requesting `PathTraced` logs a warning and falls back to the
Impostor raster path. Benchmarks expose the choice as
`--render-path flat|impostor|mesh|path-traced`, with `--light-model` and the old value
spellings kept as aliases.

## 9. Phased plan

1. **Engine RT plumbing**: extensions/features, BLAS/TLAS helpers, RT pipeline +
   SBT, storage image + blit. Exit: static frame of a hardcoded icosphere grid.
2. **Scene**: instanced icosphere BLAS/TLAS from the live interop position buffer,
   CUDA instance-writer kernel, per-frame TLAS rebuild. Exit: animated cheese,
   sun + sky, lambertian bounces.
3. **Modes & HUD**: orbit vs cube-rotation, fixed sun in both, `T` toggle, knobs,
   TLAS/trace timestamps in HUD.
4. **Benchmark integration**: CSV columns, usage text, sweep scripts, comparison
   runs vs the datoviz raster baseline.
