# Design: 64-bit particle counts — render >2^32 particles

**Date:** 2026-07-16
**Branch:** feature/remote-rendering
**Status:** Approved design, ready for implementation planning
**Builds on:** the shipped transversal `--lod` feature and the existing path-tracing BLAS chunking.

## Problem

mimir silently caps particle counts at 2^32. `point_count` is a 32-bit
`unsigned int`, so passing exactly 2^32 truncates to **0** (and 2^32+k → k),
producing a 0-particle scene that then fails deep in allocation with
`ERROR_OUT_OF_DEVICE_MEMORY` (an `IOT`/abort). This is not a memory limit:
2^32-1 ≈ 4.29 billion particles renders fine (~65 GB in phong on a 96 GB card).

mimir's purpose is to visualize *huge* particle systems, so the count must scale
to whatever GPU memory allows — across **all four light models** (`none`,
`phong`, `phong-mesh`, `path-tracing`), with **or without** LOD.

## Root cause: uint32 element counts in three layers

Buffers are already 64-bit (`allocLinear(size_t)` — a 49 GB position buffer works
today). The cap is the count *type*:

1. **Public API** (`view.hpp`): `Layout::make(unsigned int, …)`, `Layout{x,y,z}`,
   `Layout::getTotalCount()`, `AttributeDescription::size` — all 32-bit.
2. **Internal view** (`api.hpp`): `draw_count`, `instance_count` — `uint32_t`.
3. **PT engine** (`raytracing.cpp`): `bindScene(uint32_t count)`,
   `particle_count`, and the BLAS-chunk index math — `uint32_t`.

Separately, three GPU boundaries are `uint32` **by Vulkan spec**, independent of
how we type the count:

- **Raster draw:** `vkCmdDraw`/`vkCmdDrawIndexed` `vertexCount`/`instanceCount`
  and `firstVertex`/`firstInstance` are `uint32`.
- **BLAS geometry:** `primitiveCount` is `uint32` (PT already chunks around this).
- **Compute addressing:** `gl_GlobalInvocationID.x` is `uint32`, capping a 1-D
  dispatch's addressable range at 2^32 even when more workgroups are launched.

## Chosen architecture (Approach A): 64-bit total, uint32 per-operation, chunk at the boundaries

One value changes meaning: the **total** element count becomes 64-bit and flows
through the public API and the engine. Everything that is physically a
per-GPU-operation count stays `uint32`, and each Vulkan boundary that forces
`uint32` gets a loop driven by the 64-bit total. This tells the truth about the
hardware (data is 64-bit; each submission is 32-bit) and reuses the chunking
pattern PT already has. Rejected alternatives: widening *every* count field
including `draw_count` (misleading — no draw exceeds 2^32 — and still needs the
loops); PT/compute-only (leaves raster capped, violating the all-modes goal).

### 1. Public API + view types (the clean break)

- `view.hpp`: `Layout { size_t x, y, z }`, `Layout::make(size_t x, size_t y = 1,
  size_t z = 1)`, `Layout::getTotalCount() -> size_t` (currently `x*y*z` computed
  in 32-bit). `AttributeDescription::size -> size_t`.
- `api.hpp`: add `uint64_t element_count` to the internal view — the true
  particle total, and the single source of truth the chunk loops read.
  `getDrawCount()` returns `uint64_t` and sets `element_count`.
  - The existing `draw_count`/`instance_count` fields stay `uint32_t`, but only
    where they hold a genuinely bounded value: for **phong-mesh**, `draw_count`
    remains the icosphere *index* count (small, constant across chunks) and
    `instance_count` is superseded by `element_count` (the per-chunk instance
    count is derived in the loop). For **point markers**, the per-chunk
    `vertexCount` is derived from `element_count` in the loop (`instance_count`
    stays 1). No stored field holds the >2^32 total except `element_count`.
- All samples updated to pass 64-bit counts (mechanical; existing values fit).

### 2. Raster draw-chunking (`engine.cpp::drawElements`)

Replace the single `vkCmdDraw(total, …)` with a loop over chunks of ≤ `CAP` =
`UINT32_MAX` (2^32-1) vertices — the Vulkan hard max for `vertexCount`, with no
separate device limit on vertices-per-draw. Because `firstVertex` is `uint32` and
cannot express a start past 2^32, each chunk **rebinds the vertex buffer at a
64-bit byte offset** (`vkCmdBindVertexBuffers` takes `VkDeviceSize`) and draws
with `firstVertex = 0`:

```
for (uint64_t start = 0; start < total; start += CAP) {
    uint32_t n = (uint32_t)min<uint64_t>(CAP, total - start);
    // bind the position vbo at byte offset start * stride
    vkCmdDraw(cmd, n, instance_count, 0, 0);
}
```

- **none / phong** (point markers): chunk `vertexCount`, rebind vbo slot 0.
- **phong-mesh** (instanced icospheres): chunk `instanceCount`, rebind the
  per-instance buffer (binding 1) at a 64-bit offset; the icosphere template
  (binding 0) and `indexCount` are constant across chunks.
- **With `--lod`:** the loop runs once (the reduced set is ≤ N^3 < 2^32), so LOD
  remains a no-cost special case; the LOD indirect-draw paths are unchanged.

`CAP = UINT32_MAX` keeps the chunking a pure **extension**: every count that
renders today (≤ 2^32-1, including the 2^31–2^32 band) still runs as a *single*
draw, byte-for-byte unchanged, and the loop engages a second chunk only for
counts that cannot render today at all (> 2^32-1). A smaller CAP would needlessly
re-chunk already-working counts. The loop uses 64-bit `start`/`total`, so
`start += CAP` and `total - start` never overflow.

### 3. PT BLAS chunk-math → 64-bit (`raytracing.cpp`)

PT already splits particles into `ceil(count / blas_chunk_prims)` BLASes. Widen
the indexing only:

- `bindScene(VkDeviceAddress positions, uint64_t count, …)`; `particle_count`,
  `createDynamicBlasChunks(…, uint64_t count)` → 64-bit.
- `chunkPrimCount`: `uint64_t base = (uint64_t)c * blas_chunk_prims;` then
  `uint64_t rem = particle_count - base;` return `(uint32_t)min(rem,
  blas_chunk_prims)` (a chunk is ≤ blas_chunk_prims ≤ ~2^29 — fits `uint32`).
- `num_chunks` stays `uint32_t` (at 137 B particles / 2^29 chunk ≈ 256 chunks).
- `chunkAabbAddr` already computes in `VkDeviceSize` — unchanged.
- Per-BLAS `primitiveCount` stays `uint32` (each chunk is legal).

### 4. Compute passes → 64-bit grid-stride (shaders)

The LOD scatter and the PT AABB writer dispatch one thread per particle and index
by `gl_GlobalInvocationID.x`. Replace with a **bounded dispatch + 64-bit
grid-stride loop** (the pattern the CUDA sim already uses):

```
// push constant: uint64_t count
for (uint64_t i = gid; i < count; i += total_threads) { /* BDA-address element i */ }
```

- Dispatch `groupCountX = min(ceil(count / local_size), CAP_GROUPS)` so total
  threads stay ≤ 2^31 and within `maxComputeWorkGroupCount[0]`; the stride loop
  covers the remainder. BDA pointer arithmetic is already 64-bit.
- Needs `shaderInt64` for the 64-bit loop counter — already required by the LOD
  path and present on all RT-capable NVIDIA GPUs (PT needs RT hardware anyway).
- **Shaders touched:** the AABB writer and `pathtrace_lod_scatter.slang` gain the
  grid-stride loop and a 64-bit count push constant. The emit/finalize passes are
  over N^3 cells and stay 1-D (N ≤ 1625 keeps N^3 < 2^32).

### 5. Samples + simulation

- `rr-server.cu`, `benchmark_mimir.cpp`: `point_count` / `PointsParams::count` →
  `uint64_t`; positional parse reads the full 64-bit value.
- `kmodal_sim.cu`: `PointsParams::count → uint64_t`; the `d_ids` allocation
  (`sizeof(unsigned int) * count`) uses the 64-bit size. Kernels already
  grid-stride over `size_t point_count`, so no kernel-index changes.
- **Remove the `>= 2^32` guard** added earlier (obsolete); keep the `count == 0`
  reject.

### 6. Limits & error handling — memory pre-flight instead of a hard cap

Replace the fixed particle ceiling with an up-front memory estimate that rejects
with a clear message *before* Vulkan OOMs (turning the `IOT`/OOM abort into a
clean error). Estimate the dominant device allocations for the requested count:

- positions: `count × 12 B` (always),
- PT **without** LOD: `+ count × 24 B` (per-particle AABBs),
- PT **with** LOD / raster: negligible extra (LOD accumulator handled by the
  existing runtime free-VRAM check),

and reject if the estimate exceeds currently-free VRAM (queried live via
`cudaMemGetInfo`, as the LOD cap already does), reporting "needs X GB, only Y GB
free." This generalizes the LOD-accumulator check already in both samples into
one pre-flight. There is no longer a user-facing count cap — only memory. The
Vulkan per-operation `uint32` limits are handled internally by the chunk loops.

## Memory reality (sets each card's ceiling)

| path | ~bytes/particle | 96 GB card | 288 GB (B300) |
|------|-----------------|------------|----------------|
| none / phong / PT+LOD | ~12 | ~7.5 B | ~20 B |
| PT without LOD | ~36 | ~2.6 B | ~7.8 B |

This is why LOD "makes it easier": the same simulated count needs far less render
memory because only occupied-cell representatives reach the BVH / draw.

## Determinism & invariants

- Counts ≤ 2^32-1 render exactly as today: the raster loop runs one chunk (the
  draw path is unchanged — only the count *type* widened), so no existing scene
  changes.
- Counts > 2^32-1 (new territory) chunk into multiple draws; the chunk boundary is
  a pure partition of a contiguous vertex range, so the rendered cloud must show
  **no seam or gap** between chunks.
- LOD determinism (integer count + int64 centroid atomics) is unchanged; the
  grid-stride loop visits every particle exactly once, so occupied counts and
  centroids are identical to the pre-change reduction.
- `--lod 0` and small counts are behavior-preserving in every mode.

## Module boundaries

- `view.hpp` / `api.hpp`: the count-type widening and the new `element_count`.
- `engine.cpp`: raster draw-chunk loop in `drawElements`; the layout→count
  derivation; the vertex-buffer rebind-at-offset helper.
- `raytracing.cpp`: 64-bit `bindScene`/`particle_count`/chunk index math; the
  AABB-writer dispatch switches to grid-stride.
- `lod.cpp` + `pathtrace_lod_scatter.slang` + the AABB-writer shader: grid-stride
  loop, 64-bit count push constant.
- Samples (`rr-server.cu`, `benchmark_mimir.cpp`, `kmodal_sim.cu`): 64-bit count,
  memory pre-flight, guard removal.

## Testing & verification (drive-the-app; no GPU unit harness)

- **PT + LOD > 2^32** (e.g. 5 B) headless on the 96 GB card (~60 GB positions):
  the logged count is correct (no wrap), the reduction runs, a frame renders.
- **none / phong > 2^32** (~5 B, ~60 GB): the draw-chunk loop renders across
  multiple `vkCmdDraw`s with no wrap; a client receives non-blank frames.
- **PT without LOD > 2^32**: memory-bound to the B300 (154 GB at 4.3 B); verified
  there. NOT reproducible on the 96 GB card (documented limitation).
- **Regression:** a count ≤ 2^32-1 (e.g. 3 B) produces a byte-identical image to
  the pre-change build (its raster path is a single draw — only the count type
  changed); `--lod 0` and every light model are unchanged at existing counts. The
  new >2^32-1 chunked draw must render a seamless cloud (no gap at the chunk
  boundary).
- **Determinism:** repeated LOD runs at a fixed N above 2^32 give the identical
  occupied count.

## Non-goals (this change)

- Multi-GPU / out-of-core streaming for counts that exceed a single GPU's VRAM.
- Widening `draw_count`/`instance_count` semantics (they are per-chunk `uint32`
  by design).
- Removing the N ≤ 1625 LOD grid cap (a separate uint32 cell-index limit).
- 64-bit indexing for the emit/finalize passes (N^3 < 2^32 by the grid cap).
- A pixel-perfect memory estimator (the pre-flight is a conservative guard, not an
  exact allocator model).

## Success criteria

- Passing a count in (2^32, memory-limit] renders in every light model with and
  without LOD, bounded only by free VRAM — no silent wrap, no `IOT`/OOM abort.
- An over-memory count is rejected up front with a clear "needs X / Y free"
  message.
- Counts ≤ 2^32-1 are visually identical to the current build (single-draw path
  unchanged; only the count type widened); `--lod 0` and determinism preserved.
- The public API (`Layout`, `AttributeDescription::size`) is 64-bit and all
  samples compile and run against it.
