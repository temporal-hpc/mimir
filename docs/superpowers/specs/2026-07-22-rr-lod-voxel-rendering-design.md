# LOD voxel rendering — design

**Date:** 2026-07-22
**Component:** `samples/remote-rendering` (rr-server) + `lib` (engine, raytracing)
**Status:** approved (brainstorming), pending implementation plan

## Goal

Give the path-traced LOD representation a distinct **visual identity**: render each
occupied LOD cell as a solid, grid-aligned cube (voxel) instead of an inscribed sphere.
The aggregated view should read unmistakably as an *approximation* of the particle field,
not as "the same scene with fewer particles." This serves the benchmark/paper narrative
where LOD is deliberately shown as a reduced level of detail.

Default behavior is unchanged (spheres); voxel rendering is opt-in.

## Background / why this is small

The rendering machinery already exists:

- The AABB writer (`shaders/pathtrace_aabbs.slang`, `writeAabbsMain`) already emits a
  **cube AABB** (`center ± radius`) per representative.
- The intersection shader (`shaders/pathtrace.slang:287` `primitiveIntersect`) already has
  a **box/slab branch** (`:308`, selected when `sun_dir.w >= 0.5`) alongside the sphere
  branch (`:343`). The box branch does a slab test and reports a flat, face-aligned normal.
- The shape selector `sun_dir.w` is set in `engine.cpp:2537`:
  `shape_w = raytracing.voxel_boxes ? (1.f + raytracing.voxel_opacity) : 0.f`.

Today the LOD reduction path always passes `sun_dir.w = 0` (sphere). We only need to let the
LOD path opt into the existing box branch and size the box to fill the cell. **No shader
changes are required.**

## Critical separation: box *shape* vs the CA3D voxel *pipeline*

`voxel_boxes` currently gates **two** distinct things at once:

1. **Box shape** — `sun_dir.w` box branch (`engine.cpp:2537`).
2. **The entire CA3D voxel pipeline** — `recordVoxelUpdate` (`raytracing.cpp:1388`), *skipping*
   `reduceLodCompute` (`raytracing.cpp:1349` `if (lod != nullptr && !voxel_boxes)`), and the
   per-frame `voxelCompactLiving` compaction that sets `voxel_prim_count`.

This feature wants **only #1**. It must keep the normal LOD reduction pipeline
(`lod_cells > 0`: per-frame `reduceLodCompute` → AABB writer over reduced positions → BLAS
build). Therefore:

- Add a **separate** flag `lod_voxel` (engine/raytracing field, e.g. `pt_lod_voxel`) that
  affects **only** the shape selector and the LOD radius.
- The shape selector becomes:
  `shape_w = (voxel_boxes || lod_voxel) ? (1.f + opacity) : 0.f`.
- All pipeline branches keyed on `voxel_boxes` (`raytracing.cpp:1349`, `:1370`, `:1388`)
  stay **exactly as they are** — `lod_voxel` never enters them, so the LOD reduction path is
  untouched.

## Box sizing — full-cell tiling

In voxel mode the LOD sphere radius is **overridden** to half the cell edge so that
cell-center-placed boxes butt face-to-face into contiguous blocks:

```
radius = (domain_extent / grid_res) / 2      // half the LOD cell edge
```

where `grid_res` is the `--lod N` resolution and `domain_extent` is the span of the domain
the LOD grid covers along one axis (the reduction already derives the cell edge from these;
the implementation reuses that source of truth rather than re-deriving `[-1,1]` by hand).

- This **replaces** the `--size` cell-fill radius used for spheres. `--size` is **ignored**
  for the box extent in voxel mode.
- If the user passes `--size` together with `--lod-voxel`, log a one-line note that `--size`
  is ignored for voxel extent (boxes always fill the cell).
- Exact `e/2` tiling is correct for ray tracing: there is no z-fighting (hits are geometric),
  and shared faces / edges between neighbors are measure-zero. No inflation epsilon.

## Placement — auto-force cell-center

Full-cell voxels only tile cleanly on the grid lattice. Centroid placement puts each
representative at the cluster center of mass (inside the cell but off-lattice), so full-size
boxes would straddle cell boundaries and overlap neighbors into a jittered mass.

In voxel mode, **auto-force cell-center placement** (override `centroid`) and log:

```
[info] LOD voxels: placement forced to cell-center (centroid off-lattice, breaks tiling)
```

Sphere LOD keeps centroid as the default. The override only happens when `--lod-voxel` is set.

## Opacity

Opaque only for this feature: `opacity = 1.0` → `shape_w = 2.0`, which yields no transmission
rays in the integrator (`pathtrace.slang:147,176`). No `--lod-opacity` flag (YAGNI). The box
branch already supports translucency if a future change wants it.

## CLI surface

New boolean flag on `rr-server`:

```
--lod-voxel        Render LOD representatives as solid grid-aligned cubes (voxels)
                   instead of spheres. Forces cell-center placement and full-cell
                   fill (ignores --size for the box extent). Default: off (spheres).
```

Orthogonal to `--lod N` and `--lod-placement`. Off by default, so existing runs are
unchanged.

## Files touched

- `samples/remote-rendering/rr-server.cu` — parse `--lod-voxel`; help text; plumb into
  `options` (new `pt_lod_voxel` field next to `pt_lod_cells`); the `--size`-ignored and
  placement-forced log notes; force `lod_centroid = false` when voxel mode is on.
- `lib/include/private/mimir/raytracing.hpp` (or the options struct that carries
  `pt_lod_cells`) — new `lod_voxel` / `pt_lod_voxel` field.
- `lib/src/engine.cpp:2537` — `shape_w` OR with the new flag; apply the cell-edge radius
  override for the LOD build when voxel mode is on.
- The LOD radius computation site (where the sphere cell-fill radius is currently derived
  from `--size`) — override to `e/2` in voxel mode.

**No shader changes.**

## Testing

Headless verification (mirrors the render_ms fix run):

1. Server: `rr-server ... --light-model path-tracing --lod 64 --lod-voxel ...` at a small N.
2. Assert startup logs report: box/voxel mode active, placement forced to cell-center, and
   the LOD radius equal to `e/2` for the chosen grid.
3. Run the headless client for a bounded frame count; assert `render_ms` is ~unchanged versus
   the sphere run (trace cost is negligible; the frame is dominated by the LOD reduction).
4. Save `rr-client.ppm` and visually confirm cubes (flat-shaded faces, contiguous blocks)
   versus round blobs in the sphere run.
5. Regression: a run **without** `--lod-voxel` still renders spheres, and a CA3D
   `voxel_boxes` run is unaffected (the two flags are independent).

## Non-goals

- Translucent / opacity-controlled LOD voxels.
- Triangle-mesh cubes (hardware triangle intersection) — much larger BLAS, no benefit given
  the trace is <1 ms; explicitly out of scope.
- Any change to the LOD reduction algorithm, the CA3D `voxel_boxes` pipeline, or sphere LOD
  behavior.
