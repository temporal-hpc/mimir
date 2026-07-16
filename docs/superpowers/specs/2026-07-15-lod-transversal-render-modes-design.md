# Design: transversal `--lod` across all render modes

**Date:** 2026-07-15
**Branch:** feature/remote-rendering
**Status:** Approved design, ready for implementation planning
**Builds on:** `2026-07-14-lod-grid-aggregation-design.md`, `2026-07-14-lod-centroid-bda-accumulator-design.md` (the shipped path-tracing-only `--lod`).

## Problem

`--lod N` currently lives entirely inside the path-tracing pipeline
(`raytracing.cpp`): its scatter/emit passes aggregate particles directly into
BVH **AABB primitives**. The raster light models (`none`, `phong`,
`phong-mesh`) draw the full particle buffer directly and are unaffected by
`--lod`. LOD is really *data reduction* — how many things to draw — which is
orthogonal to shading, so it should apply to every light model; the user picks
`--light-model` and `--lod` independently.

Secondary wart: in LOD mode the sphere radius is `coverage * cellSize / 2`
(cell-derived), which overrides `--size` entirely — so `--size` appears dead
when `--lod` is on.

## Goal

1. Make `--lod` **transversal**: `none`, `phong`, `phong-mesh`, and
   `path-tracing` all render the reduced particle set.
2. Keep `--size` meaningful under `--lod` (a multiplier on the cell-fill radius,
   `none` uses it as pixel size directly).
3. Unify into **one** LOD codepath (no parallel raster/PT aggregation).
4. Preserve determinism, the `--lod 0` no-op, and the existing VRAM/uint32 cap.

## Chosen architecture

Extract LOD from `raytracing.cpp` into a **shared pre-render reduction stage**
(new module `lib/src/lod.cpp` + `lib/include/private/mimir/lod.hpp`) that runs
each frame when `pt_lod_cells > 0`, regardless of light model, and produces a
**reduced particle set** instead of AABBs:

- **Reduced-position buffer:** compact list of occupied-cell centroids
  (`float3`), sized `min(N^3, P) * 12 B`, usage
  `VERTEX_BUFFER | SHADER_DEVICE_ADDRESS | STORAGE` (consumed both as a raster
  vertex buffer and, via BDA, by the path-tracer's AABB writer).
- **Count buffer:** `uint` occupied-cell count (the existing `lod_counter`
  role). Single source of truth for the reduced count.
- The scatter pass is unchanged (count + int64 centroid sum). The emit pass
  changes from "write AABB per occupied cell" to "write **centroid position**
  per occupied cell" into the reduced-position buffer (still appends via the
  `globalCount` atomic).

### Consumers

- **Path-tracing:** no longer emits AABBs in the LOD pass. Instead it runs its
  **existing per-particle AABB writer** over the reduced-position buffer with
  `count = occupied` (read back as today), then builds BLAS/TLAS unchanged. So
  the AABB writer's input buffer switches from the interop positions (`--lod 0`)
  to the reduced positions (`--lod N`); everything downstream is identical.
- **Raster (`none`/`phong`/`phong-mesh`):** the marker instanced draw binds the
  reduced-position buffer as its vertex buffer (`view->vbo`) and uses an
  **indirect draw** whose varying count comes from a GPU indirect-args buffer
  (no host readback). See Raster integration.

### Frame flow (LOD active)

`recordLodReduction(cmd)` runs early in the frame command buffer (after the sim
has produced the live interop positions, before raster `drawElements` / PT
`recordUpdateScene`): clear accumulators -> scatter -> emit (writes reduced
positions + `globalCount`). Then the mode-specific consumer runs. A barrier
serializes the reduction's writes before the draw/AABB-writer reads. (v1 uses a
single reduced buffer + barrier — see Non-goals for the multi-buffering perf
follow-up.)

## Raster integration (indirect draw)

Positions are a **vertex buffer** today (`engine.cpp:2122`). Per mode the LOD
count lands in a different draw field:

- **`none` / `phong`** (`vkCmdDraw(vertexCount=P, instanceCount=1)`): LOD makes
  **vertexCount = occupied**. Use `vkCmdDrawIndirect` reading a
  `VkDrawIndirectCommand{ vertexCount, instanceCount=1, firstVertex=0,
  firstInstance=0 }`.
- **`phong-mesh`** (`vkCmdDrawIndexed(indexCount=icosphere, instanceCount=P)`):
  LOD makes **instanceCount = occupied**. Use `vkCmdDrawIndexedIndirect`
  reading a `VkDrawIndexedIndirectCommand{ indexCount=icosphere, instanceCount,
  firstIndex=0, vertexOffset=0, firstInstance=0 }`.

**Indirect-args construction** (mode-agnostic LOD + a tiny render-side step): the
LOD stage produces only `globalCount`. A small "build indirect args" compute
dispatch (1 thread) writes the correct `VkDraw*IndirectCommand` into an indirect
buffer from `globalCount` plus the fixed fields (known on the host: `1` for the
point-mode instanceCount, the icosphere `indexCount` for mesh). The varying
field offset + the fixed fields are passed as push constants, so one generic
finalize shader serves both command layouts. The draw then uses
`vkCmdDraw*Indirect` from that buffer. Indirect buffer usage:
`INDIRECT_BUFFER | STORAGE | TRANSFER_DST`.

When `pt_lod_cells == 0`, the raster path is exactly as today (direct
`vkCmdDraw*` over the interop positions, full count) — no indirect, no reduced
buffer.

## `--size` behavior

- **Lit (`phong`/`phong-mesh`/`path-tracing`):**
  `radius = cellFill * (default_size / LOD_REFERENCE_SIZE)`, where
  `cellFill = coverage * cellSize / 2` and `LOD_REFERENCE_SIZE` is the light
  model's default `--size` (so the default fills the cell = today's opaque look;
  scaling `--size` up/down makes blobs chunkier/thinner). Never dead; behaves
  identically at any `--lod N` because it is cell-relative. The lit marker size
  (`engine.cpp:2232`) and the PT AABB radius both use this value.
- **`none`:** `--size` stays the pixel point size on the centroid — no world
  radius / cell-fill concept in flat 2D. LOD only reduces the point count.

## Determinism & invariants

- Determinism unchanged: same integer count + int64 centroid atomics; the emit
  `globalCount` atomic only affects append order, not the count/centroids.
- The SAME reduced set (positions + count) feeds raster and PT, so switching
  `--light-model` at a fixed `--lod N` shows a consistent scene.
- `--lod 0` is byte-for-byte the current behavior in every mode.
- The cap is unchanged (VRAM-scaled, clamped to N <= 1625 for the uint32
  cell-index limit).

## Module boundaries

- New `lib/include/private/mimir/lod.hpp` + `lib/src/lod.cpp`: owns the
  accumulator/reduced/count buffers, the scatter/emit/finalize compute
  pipelines, `recordLodReduction(cmd)`, and the reduced-count readback helper.
  Independent of the render mode.
- `raytracing.cpp`: drops its internal scatter/emit; `recordUpdateScene` (LOD
  path) becomes "run the AABB writer over the reduced positions + count from the
  LOD module, then build". The centroid/int64/BDA machinery moves to `lod.cpp`.
- `engine.cpp`: calls `recordLodReduction` early each frame when LOD is on;
  raster `drawElements` binds the reduced vbo + uses indirect draw when LOD is
  on; wires the `--size` multiplier.

## Testing & verification (drive-the-app; no GPU unit harness)

- **Per mode:** run each of `none`, `phong`, `phong-mesh`, `path-tracing` with
  `--lod 32` at 2^20 and confirm it renders a reduced (blobby) cloud and logs
  the same occupied-cell count across modes; and that `--lod 0` renders the full
  cloud unchanged in each mode.
- **Determinism:** repeated runs at fixed N give the identical occupied count.
- **`--size`:** with `--lod` on in a lit mode, sweeping `--size` visibly changes
  blob thickness (not dead); at default `--size` the cloud is opaque.
- **Consistency:** the occupied count is identical across all four modes at the
  same N.
- **Regression:** `--lod 0` unchanged in every mode; path-tracing `--lod N`
  still emits the same occupied count as before this change (13689 @ --lod 32 /
  2^20) — the reduction result must be identical after moving it to `lod.cpp`.

## Non-goals (this change)

- Multi-buffering the reduced buffer for raster frame overlap (v1 uses a single
  buffer + barrier; a perf follow-up if raster stalls matter).
- Per-cell variable representative size (uniform cell-fill * multiplier only).
- Sparse-hash accumulator / view-adaptive LOD (prior non-goals stand).
- Indirect BLAS build for PT (PT keeps its count readback).

## Success criteria

- `--lod N` visibly reduces the cloud in all four light models, user-selectable
  independently of `--light-model`.
- `--size` is live under `--lod` in lit modes (multiplier on cell-fill) and is
  the pixel size in `none`.
- One LOD codepath (the shared module); no duplicated aggregation.
- Determinism, `--lod 0` no-op, and the VRAM/uint32 cap all preserved; PT's
  occupied count is unchanged from the shipped version.
