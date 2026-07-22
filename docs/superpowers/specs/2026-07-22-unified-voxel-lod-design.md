# Unified voxel-LOD — design

**Date:** 2026-07-22
**Component:** `samples/remote-rendering` (rr-server) + `lib` (engine, raytracing, shaders)
**Status:** approved (brainstorming), pending implementation plan
**Supersedes:** the PT-only `--lod-voxel` feature (commits 7aa0478 / e4ccbf5 / 1acb54f) — that flag is
removed and replaced by the unified shape knob below.

## Goal

Make **voxels the default, universal representation of LOD** across all *lit* light models —
`phong` (Sphere3D impostor), `phong-mesh` (SphereMesh), and `path-tracing` — so the reduced level of
detail always reads as honest aggregated occupancy (a solid, grid-aligned voxel field) rather than
"the same scene with fewer, smaller spheres." `none` (flat pixel-point splats) is excluded — there is
no 3D surface to voxelize.

This aligns the render with what `--lod` already *is*: its own help calls it "an N^3 **voxel grid**,
one representative per occupied cell." Rendering that as voxels is the honest depiction.

## Decisions (locked in brainstorming)

- **Default = voxel** for all lit LOD. Opt out with `--lod-shape sphere`.
- **Full-cell tiled** voxels (half-extent = half the cell edge → contiguous blocks), same as the
  PT feature already shipped.
- **Cell-center placement is auto-forced** whenever voxels render (centroid is off-lattice and breaks
  tiling). Centroid is reachable only via `--lod-shape sphere`.
- **Remove `--lod-voxel`** entirely; the new knob is `--lod-shape sphere|voxel` (default `voxel`).
- **Raster voxels via an instanced unit-cube mesh** (reuse the SphereMesh instancing path), not a new
  cube-impostor shader.

## 1. The shape knob (replaces `--lod-voxel`)

- Generalize `ViewerOptions::pt_lod_voxel` → **`bool lod_voxel = true`** (default voxel). It is consumed
  by *both* the PT path and the new raster path. `RayTracingContext::lod_voxel` remains and is set from
  it (now defaulting on).
- rr-server: **remove** the `--lod-voxel` flag (and its `lod_voxel_render` plumbing / the
  `light_model != PathTracing` warning added for it). Add **`--lod-shape sphere|voxel`** (default
  `voxel`): `voxel` → `lod_voxel = true`, `sphere` → `lod_voxel = false`. Any other value is an error
  (like `parseLightModel`).
- The "force cell-center + ignore `--size`" behavior moves to apply whenever `lod_voxel` is active under
  a lit model with `--lod` (see §4). Under `none`, `lod_voxel` is irrelevant (points); `--lod-shape
  voxel` there is a silent no-op.

## 2. Path tracing (already implemented)

The box branch (`shaders/pathtrace.slang:308`, selected by `sun_dir.w >= 0.5`) and
`LodContext::voxelHalfExtent()` already exist. The only change is that `lod_voxel` now defaults to true,
so PT LOD renders voxels unless `--lod-shape sphere` is passed. No shader or pipeline change for PT.

## 3. Raster (the new work): instanced cube mesh

Reuse the existing instanced-mesh path (`ensureSphereMesh` at `engine.cpp:1094`; the per-instance
centers in vbo binding 1, drawn with `instanceCount` = occupied cells, `engine.cpp:394-400`). Add a
cube template and route lit LOD through it when `lod_voxel`:

- **`ensureCubeMesh()`** — build a **24-vertex unit cube** (`[-1,1]^3` corners, 4 vertices per face so
  each face carries a flat normal) + **36 indices** (12 triangles), lazily, mirroring `ensureSphereMesh`.
  New buffers `cube_vbo` / `cube_ibo` (+ their index count), alongside `sphere_vbo`/`sphere_ibo`.
- **Add an explicit per-vertex normal attribute** (location 2) to the instanced-mesh vertex input
  layout. The icosphere template sets `normal = position` (unit sphere); the cube template sets
  `normal = face normal`. (This means the icosphere template VBO gains a normal channel equal to its
  positions; the pipeline's vertex-input description gains one attribute + matching stride.)
- **`shaders/marker_mesh.slang` `vertexMain`:** change
  `float3 n = normalize(in_local); float3 vpos = center_view.xyz + n * view.default_size;`
  to read the raw local offset and the explicit normal:
  `float3 vpos = center_view.xyz + in_local * view.default_size; vnormal = normalize(in_normal);`
  This is **identical** for the icosphere (its `in_local` is already unit length, and `in_normal ==
  in_local`), and **correct** for the cube (raw corner offset builds a real cube; per-face normals give
  flat-shaded faces). Lighting is the Blinn-Phong already fixed in this shader.
- **Sizing:** the cube uses `size = lod->voxelHalfExtent()` (cell-edge/2 → tiling); the icosphere keeps
  `lod->sphereRadius(default_size)`. This size is what the vertex shader multiplies `in_local` by
  (fed through the same `view.default_size` UBO field the raster LOD path already sets per-view at
  `engine.cpp:3243`-ish — set it to `voxelHalfExtent()` when voxel).
- **Pipeline selection under `--lod` (lit models):**
  - `lod_voxel` (default) → **cube mesh** (new path) — regardless of the marker's Sphere3D/SphereMesh
    render mode.
  - `--lod-shape sphere` + Sphere3D (`phong`) → sphere impostor (current).
  - `--lod-shape sphere` + SphereMesh (`phong-mesh`) → icosphere mesh (current).
  - `none` → points (current), shape ignored.

## 4. Placement

Cell-center placement is auto-forced whenever voxels render. Put the force in the **engine** LOD setup
(where `options.pt_lod_cells` / `options.lod_centroid` are consumed, `engine.cpp:237`), not only in
rr-server, so any caller gets consistent behavior: if `lod_voxel && pt_lod_cells > 0`, force
`lod_centroid = false` and log once. rr-server keeps a user-facing note (it already prints one).
Centroid is honored only when `--lod-shape sphere`.

## 5. Performance

Neutral-to-faster for phong versus the sphere impostor:
- No per-fragment ray-sphere (`sqrt` / `normalize` gone), no shader depth write → **early-Z culls
  overdraw** (the mesh path's own documented advantage, `marker_mesh.slang:5-6`).
- Flat per-face normals; only 12 triangles per cell over the **LOD-reduced** count.
- The frame stays dominated by the LOD reduction, not the draw (same as PT).

## Files touched

- `lib/include/public/mimir/options.hpp` — rename `pt_lod_voxel` → `lod_voxel`, default `true`.
- `lib/include/private/mimir/engine.hpp` — declare `ensureCubeMesh()`, `cube_vbo`/`cube_ibo`/index count.
- `lib/src/engine.cpp` — `ensureCubeMesh`; add the normal attribute to the instanced-mesh vertex input
  and set the icosphere's normal channel; cube-vs-sphere template + size selection in the raster LOD
  draw; force cell-center when `lod_voxel`; set `raytracing.lod_voxel = options.lod_voxel`.
- `shaders/marker_mesh.slang` — normal attribute (location 2) + raw-offset / explicit-normal change.
- `samples/remote-rendering/rr-server.cu` — remove `--lod-voxel`; add `--lod-shape sphere|voxel`
  (default voxel); update help; keep the cell-center / `--size`-ignored notes for voxel mode.
- PT path (`raytracing.cpp`, `pathtrace.slang`) — unchanged (already implemented).

## Testing (headless, integration-style)

Mirror the earlier voxel/lighting verification (background server, headless `rr-client`, log grep +
saved PPM):

1. **Default is voxel:** `--light-model path-tracing --lod 64` (no shape flag) → emit log reports voxel
   half-extent; PPM shows cubes. Same for `--light-model phong --lod 64` → cube-mesh render (PPM shows
   flat-faced cubes, not round spheres).
2. **Opt-out:** `--lod-shape sphere` under phong → sphere impostors (round), placement stays as given
   (centroid honored); under PT → sphere branch.
3. **All lit models voxelize:** phong, phong-mesh, path-tracing each render cubes by default; visually
   compare PPMs.
4. **`none` unaffected:** `--light-model none --lod 64 [--lod-shape voxel]` → points, no cubes, no
   cell-center force.
5. **Placement forced:** voxel + `--lod-placement centroid` → log shows cell-center selected.
6. **`--lod-voxel` is gone:** passing it is an unknown-flag path (documents the removal).
7. **Perf sanity:** phong `--lod` voxel `render_ms` is not worse than the sphere-impostor LOD run.

## Non-goals

- A cube-impostor fragment shader (the instanced cube mesh is simpler and faster at LOD counts).
- Voxelizing `none`.
- Any change to the LOD reduction algorithm or to non-LOD rendering.
- Retiring `phong-mesh` (separate decision; the cube mesh reuses its instancing but the icosphere path
  stays for `--lod-shape sphere` + phong-mesh and for non-LOD phong-mesh).
