# Handoff: continuing the particles-kmodal-3d path-tracing work on another machine

Written 2026-07-02 on the workstation, for picking up on the laptop (RTX GPU) after a
pull. Delete this file once the context is absorbed. Companion doc: DESIGN_pathtracing.md
(the accepted design — read it first).

## Where we are

particles-kmodal-3d is finished as a raster benchmark pair and the path-tracing feature has just
been started (public API only — no RT rendering code exists yet).

Done and tested on the workstation (RTX 3090 Ti):

- `samples/particles-kmodal-3d/`: benchmark_mimir + benchmark_datoviz share `kmodal_sim.cu`.
  K-modal gaussian init (`--k`, `--epsilon`), mean-reverting OU walk so the cheese
  shape persists (verified: cluster stddev stays ~epsilon after 5000 steps), cluster
  centers drawn over the FULL cube so wall-adjacent blobs get sliced flat by the
  clamp (with seed 12345/k 8 four blobs slice, one shows a cube edge).
- `shaders/marker_flat.slang` replicates datoviz's disc SDF exactly (antialias = 1 px
  margin) so `--size N` renders the same pixels in both benchmarks.
- benchmark_datoviz `--marker-mode 1` uses lit `dvz_sphere` impostors, size passed as
  size/50 NDC diameter == mimir's size/100 world radius.
- CSV columns `k`,`epsilon` were inserted after `seed` in BOTH benchmarks (update any
  parsing scripts).

Just written, **compiles NOT yet verified** (the rebuild was interrupted to push):

- `lib/include/public/mimir/options.hpp`: two orthogonal axes,
  `enum class Shading { Unlit, Phong, PathTraced }` + `enum class Geometry { Sprite, Impostor,
  Mesh }`, as `ViewerOptions::shading` / `::geometry` (defaults Phong + Impostor = historical
  look). This was one conflated enum until 2026-08: `LightModel { None, Phong, PhongMesh,
  PathTracing }`, briefly `RenderPath`.
- `lib/src/engine.cpp` `createView` (~line 800): resolves the instance axes onto
  `MarkerOptions::render_mode` (None→Flat2D, Phong→Sphere3D, PathTracing→warn +
  Sphere3D fallback). `MarkerOptions::render_mode` is now engine-managed (comment in
  view.hpp says so).
- `lib/src/device.cpp` + `device.hpp`: `getRayTracingExtensions()`,
  `supportsRayTracing(gpu)` (extension + feature query), and `createLogicalDevice`
  now enables VK_KHR_acceleration_structure / ray_tracing_pipeline /
  deferred_host_operations + bufferDeviceAddress feature whenever the GPU supports
  them (logged at info level).

## Immediate next steps (in order)

1. ~~**Rebuild the library from zero and fix compile fallout**~~ ✅ DONE on the laptop
   (RTX 4090, CUDA 13.2, gcc-14). The RT feature structs used C++ designated
   initializers which tripped `-Werror=missing-field-initializers` under gcc-14 —
   converted both the `supportsRayTracing` query block and the `createLogicalDevice`
   enable block in device.cpp to the `Struct x{}; x.field = …;` style used elsewhere in
   the file. Library builds clean.
2. ~~**Update both benchmarks to `--shading` + `--geometry`**~~ ✅ DONE and verified (originally
   `--light-model none|phong|path-tracing`, still accepted as an alias along with `--render-path`).
   benchmark_mimir sets `opts.shading`/`opts.geometry`, no longer touches
   marker_opts.render_mode (engine-managed), size none→px / phong,pt→/100. Both
   benchmarks parse `none|phong|path-tracing`; datoviz path-tracing prints the raster
   baseline message and exits 1. Verified windowed: mimir none≈2050 fps (flat discs),
   phong≈400 fps (lit spheres); datoviz none/phong both render.
   FIXED: the intermittent garbage `graphics_time` (~1.78e9 s) was a
   `GraphicsMonitor::stopFrameWatch` bug — a swapchain recreation (resize/OUT_OF_DATE/
   SUBOPTIMAL, common on Wayland) rebuilds the monitor mid-frame at engine.cpp:1632,
   resetting frame_start to a default TimePoint so stopFrameWatch computed now()-epoch.
   metrics.cpp now skips a frame sample when frame_start is unset. Verified 10/10
   clean runs (was ~1 in 4 corrupt).
3. **Path tracing phases 1-4** — NOT STARTED. As laid out in DESIGN_pathtracing.md §9
   (engine RT plumbing → instanced icosphere BLAS/TLAS from interop positions →
   orbit/cube-rotation modes with fixed sun → benchmark CSV integration).

## Machine gotchas (workstation-specific knowledge you'll otherwise lack)

- Multi-GPU machines: datoviz auto-picks the "best" GPU which may not drive the
  display; benchmark_datoviz self-defaults `DVZ_GPU=0` (setenv, non-overwriting).
  samples/ca and samples/nbody-datoviz do NOT have this default yet.
- The mimir shader path resolves next to the executable; particles-kmodal-3d's CMakeLists has a
  POST_BUILD copy step, but if you edit shaders without relinking, re-copy manually
  (`cmake -E copy_directory shaders <bin>/shaders`).
- Release builds silence spdlog; `SPDLOG_LEVEL=err` (or info) re-enables — essential
  for seeing slang shader compile errors, which otherwise appear as a segfault in
  ShaderBuilder::compileModule.
- `exit(instance)` MUST be called before `destroyInstance` or the process hangs.
- Benchmarks: Ctrl+W closes the window cleanly; run from
  `samples/particles-kmodal-3d/build/bin/`; smoke test:
  `SPDLOG_LEVEL=err ./benchmark_mimir 1280 720 500000 12345 240`.
