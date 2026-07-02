# Design: Vulkan Path Tracing for the points3d Scene

Status: **Accepted — implementation started** (see HANDOFF.md for exact progress)
Author: Cristobal (drafted with Claude)
Last updated: 2026-07-02

## 1. Goal

Path-trace the points3d "cheese" scene (k-modal gaussian point clusters in the [-1,1]³
cube, mean-reverting walk) with Vulkan ray tracing, accelerated by RT cores when
available. points3d completes the benchmark matrix as the render-bound workload:

| sample | compute | transfer | render |
|--------------------|---------|----------|--------|
| nbody | high | low | low |
| cellular automata | low | high | low |
| points3d (PT) | low | low | **high** |

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

### 8.1 Public API: `LightModel`

The shading choice is exposed to the programmer as an instance-wide light model
(`lib/include/public/mimir/options.hpp`):

```cpp
enum class LightModel { None, Phong, PathTracing };
struct ViewerOptions { ...; LightModel light_model = LightModel::Phong; ... };
```

- `None`        → unlit raster; markers draw as flat 2D point-sprite discs.
- `Phong`       → lit raster; markers draw as ray-sphere impostors (historical default).
- `PathTracing` → the path-traced path of this document.

`MarkerOptions::render_mode` is now engine-managed: `createView` derives it from the
instance's light model (None → Flat2D, Phong → Sphere3D); programs should not set it
directly anymore. Until phases 1–3 land, requesting `PathTracing` logs a warning and
falls back to Phong raster. Benchmarks expose the choice as
`--light-model none|phong|path-tracing`.

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
