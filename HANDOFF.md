# Handoff: continuing the points3d path-tracing work on another machine

Written 2026-07-02 on the workstation, for picking up on the laptop (RTX GPU) after a
pull. Delete this file once the context is absorbed. Companion doc: DESIGN_pathtracing.md
(the accepted design — read it first).

## Where we are

points3d is finished as a raster benchmark pair and the path-tracing feature has just
been started (public API only — no RT rendering code exists yet).

Done and tested on the workstation (RTX 3090 Ti):

- `samples/points3d/`: benchmark_mimir + benchmark_datoviz share `points3d_sim.cu`.
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

- `lib/include/public/mimir/options.hpp`: new `enum class LightModel { None, Phong,
  PathTracing }` + `ViewerOptions::light_model` (default Phong = historical look).
- `lib/src/engine.cpp` `createView` (~line 800): maps instance light_model onto
  `MarkerOptions::render_mode` (None→Flat2D, Phong→Sphere3D, PathTracing→warn +
  Sphere3D fallback). `MarkerOptions::render_mode` is now engine-managed (comment in
  view.hpp says so).
- `lib/src/device.cpp` + `device.hpp`: `getRayTracingExtensions()`,
  `supportsRayTracing(gpu)` (extension + feature query), and `createLogicalDevice`
  now enables VK_KHR_acceleration_structure / ray_tracing_pipeline /
  deferred_host_operations + bufferDeviceAddress feature whenever the GPU supports
  them (logged at info level).

## Immediate next steps (in order)

1. **Rebuild the library from zero and fix any compile fallout** of the LightModel /
   device.cpp changes: `./mimir-build-from-zero.sh --gcc 14 --remote` (drop `--gcc 14`
   if not on Arch). Watch for: the RT feature structs use designated initializers
   (zero-fill is intended); `checkAllExtensionsSupported` is forward-declared in
   device.cpp above its use.
2. **Update both benchmarks to `--light-model none|phong|path-tracing`** replacing
   `--marker-mode` (this is NOT done yet — benchmark_mimir still sets
   `marker_opts.render_mode` directly, which the engine now OVERRIDES from the default
   Phong light model, so until this step the benchmark's flat mode is broken!):
   - benchmark_mimir: set `opts.light_model` from the flag; stop touching
     marker_opts.render_mode; size mapping: none → pixels, phong/path-tracing → /100.
   - benchmark_datoviz: none → dvz_marker discs, phong → dvz_sphere lit impostors,
     path-tracing → print "datoviz is the raster baseline, cannot path trace" + exit.
   - Rebuild: `./samples-build-from-zero.sh --sample points3d --gcc 14`. Verify
     `--light-model none` gives identical-size discs in both, `phong` gives lit spheres.
3. **Path tracing phases 1-4** as laid out in DESIGN_pathtracing.md §9 (engine RT
   plumbing → instanced icosphere BLAS/TLAS from interop positions → orbit/cube-rotation
   modes with fixed sun → benchmark CSV integration).

## Machine gotchas (workstation-specific knowledge you'll otherwise lack)

- Multi-GPU machines: datoviz auto-picks the "best" GPU which may not drive the
  display; benchmark_datoviz self-defaults `DVZ_GPU=0` (setenv, non-overwriting).
  samples/ca and samples/nbody-datoviz do NOT have this default yet.
- The mimir shader path resolves next to the executable; points3d's CMakeLists has a
  POST_BUILD copy step, but if you edit shaders without relinking, re-copy manually
  (`cmake -E copy_directory shaders <bin>/shaders`).
- Release builds silence spdlog; `SPDLOG_LEVEL=err` (or info) re-enables — essential
  for seeing slang shader compile errors, which otherwise appear as a segfault in
  ShaderBuilder::compileModule.
- `exit(instance)` MUST be called before `destroyInstance` or the process hangs.
- Benchmarks: Ctrl+W closes the window cleanly; run from
  `samples/points3d/build/bin/`; smoke test:
  `SPDLOG_LEVEL=err ./benchmark_mimir 1280 720 500000 12345 240`.
