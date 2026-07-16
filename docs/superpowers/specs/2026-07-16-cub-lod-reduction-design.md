# Native-CUDA LOD Reduction — Design Spec

**Date:** 2026-07-16
**Branch:** feature/remote-rendering
**Status:** design (revised)

## REVISION (supersedes the CUB approach below)

Originally this planned a **CUB** fast path. Dropped: CUB's `DeviceRunLengthEncode`/`ReduceByKey`
take an `int` item count, capping it at `INT32_MAX` (~2.147 B) particles — disqualifying for
HPC-scale runs (billions to tens of billions). Reframing from the B300 measurements: the reduction
was slow because of the **Vulkan compute path** (B300 ran it ~190× slower than CUDA; the sim does
300 M in 1.8 ms), **not** atomics (workstation Vulkan atomics were 25–57 ms). So the fix is "run the
reduction in **native CUDA**" — achievable with our **own kernels**, which also scale past 2³²
trivially via `size_t` grid-stride (exactly like the sim), with no `int` cap.

**Approach (this spec):** port the proven scatter+emit (currently Vulkan slang) to native CUDA
`.cu` kernels — `size_t`-indexed, no external library:
- **cell-center:** benign non-atomic occupancy store (zero atomics; already validated in the Vulkan
  shader).
- **centroid:** `atomicAdd` u32 count + 3× `atomicAdd` u64 **int64 fixed-point** position sum (kept
  integer so the sum is order-independent → run-to-run deterministic, matching today's behavior).
- **emit/compact:** one thread per N³ cell; occupied cells append their representative via a global
  `atomicAdd` slot counter (the occupied count). SET is deterministic; slot order is not (unchanged).

Everything below about **interop, selection, PT/raster integration, determinism, testing** still
applies, with "CUB pipeline" replaced by "custom CUDA scatter+emit kernels" and these changes:
- **No `INT32_MAX` cap; scales past 2³².**
- **Selection** is not about CUB scratch: the custom path holds the **same N³ accumulator as the
  Vulkan path** (count u32 + optional 3× u64 sum), so if LOD was allowed at all it fits. The CUDA
  path becomes the **primary** path when CUDA/interop is available (always, for mimir); the Vulkan
  reduction is retained as a fallback/reference, selectable via `MIMIR_LOD_NO_CUDA=1`.
- **Centroid** keeps int64 fixed-point (determinism), not CUB float reduce.
- Follow-up (not now): if centroid atomic contention is still the bottleneck on the B300 after this
  port, add a contention-free custom reduction (counting/radix sort by cell) — measured first.

---

_(original CUB design retained below for reference; the CUB pipeline section is replaced by the
custom-kernel approach above)_

## Problem

The transversal `--lod` feature reduces N particles into an N³ voxel grid, emitting one
representative per occupied cell. The reduction is a per-frame Vulkan compute pass
(`pathtrace_lod_scatter.slang` + `pathtrace_lod_emit.slang`): scatter each particle into its
cell (atomic count + optional int64 position-sum), then compact occupied cells.

At huge particle counts this reduction dominates the frame. Measured (path-tracing, `--lod 256`,
`--steps-per-frame 1`):

| GPU | count | placement | reduction (`lod ms`) |
|---|---|---|---|
| RTX PRO 6000 (workstation) | 300 M | centroid | 57 ms |
| RTX PRO 6000 | 300 M | cell-center (benign store) | 25 ms |
| B300 (datacenter) | 300 M | cell-center (benign store) | **~350 ms** |
| B300 | 1 B | centroid | ~470 ms |
| B300 | 1 B | cell-center | ~380 ms |

Two findings:
1. On the workstation, atomics dominate: dropping the int64 sum atomics (cell-center) gives ~2.3×,
   and dropping the count atomic (benign store) gives more.
2. **On the B300 the Vulkan compute path itself is the bottleneck, not atomics.** The B300 runs a
   CUDA kernel over the same 300 M particles (the sim step) in **1.8 ms**, but the Vulkan reduction
   over the same 300 M takes **~350 ms** — ~190× slower, and atomic-free (the benign-store
   cell-center still costs ~350 ms there). The B300 is a compute-class GPU with a stripped/emulated
   graphics stack (`0 RT cores → software BVH`, NVENC unsupported); its Vulkan compute is a
   second-class path, its CUDA path is the fast one.

## Goal

Add a **CUDA/CUB reduction fast path** that computes the occupied-cell set + representatives in
native CUDA (bypassing the slow Vulkan compute path), selected at runtime when its scratch fits
free VRAM, with the existing Vulkan reduction retained as the universal fallback. Optimize for
speed on big-VRAM GPUs (B300); remain correct (via fallback) on smaller GPUs.

Expected: the reduction drops from hundreds of ms to single-digit ms on the B300 (CUDA does 300 M
in ~1.8 ms), and is faster than the atomic scatter on workstation GPUs for the centroid case.

## Non-Goals

- Chunked/out-of-core CUB sort for GPUs where the sort scratch does not fit — those fall back to
  the Vulkan path (still correct, just the old speed). May be added later.
- Changing the LOD grid model, the `N ≤ 1625` uint32 cap, or the emit/representative semantics.
- Multi-GPU.

## Selection & Fallback

Per session, at LOD init, when `--lod > 0`:

1. Compute the CUB fast-path VRAM need for the configured particle count:
   `sort_keys(2× u32) + sort_values(2× u32) + reduced outputs + cub_temp` (query `cub_temp` via the
   CUB API with a null temp pointer). Add a safety margin.
2. If it fits in currently-free VRAM (via `cudaMemGetInfo`) **and** CUDA/CUB is available: use the
   **CUB fast path** for the reduction (both placements).
3. Else: use the **existing Vulkan reduction** (atomic scatter for centroid; benign store for
   cell-center). Unchanged, correct on every GPU.

Log the chosen path, e.g. `LOD reduction: CUB (CUDA) fast path` or `LOD reduction: Vulkan scatter
(CUB scratch <X> GB would not fit free VRAM)`.

The CUB fast path serves **both** placements:
- **cell-center**: emit the geometric center of each unique occupied cell.
- **centroid**: emit `sum(positions)/count` per unique occupied cell (exact float, no fixed-point).

## CUB Pipeline (new `lib/src/lod_cub.cu`)

Per reduction (all on one CUDA stream), inputs are the interop position buffer (device ptr,
tightly packed float3, stride 12 B) + `count` (uint64) + `gridN`:

1. **key kernel:** `cell_id[i] = linearize(clamp(pos[i]))` → `u32` (256³ = 2²⁴; `N ≤ 1625` ⇒
   `N³ < 2³²`, fits u32). Also fill `idx[i] = i` (u32) — but see step 2.
2. **sort:** `cub::DeviceRadixSort::SortPairs(cell_id, idx)` (8-byte pairs; cheaper to move than
   float3). Bit range limited to `ceil(log2(N³))` to cut radix passes.
3. **per-cell reduce:**
   - counts + unique cells: `cub::DeviceRunLengthEncode::Encode(sorted_cell_id)` → `unique_cells`,
     `run_lengths` (= counts), `num_occupied`.
   - centroid sums (centroid mode only): `cub::DeviceReduceByKey(sorted_cell_id, gather(pos,
     sorted_idx), float3-add)` → per-cell position sums, aligned with `unique_cells`.
4. **finalize kernel:** one thread per occupied cell → write the compacted reduced-position list:
   - cell-center: center from `unique_cells[k]` (delinearize).
   - centroid: `sum[k] / count[k]`.
   `num_occupied` (clamped to `max_cells`) is the occupied count PT needs for the BLAS and raster
   needs for the indirect draw.

Buffers (keys, values, sorted outputs, RLE/reduce outputs, cub_temp) are allocated **once** at
init (sized for the configured count) and reused every frame.

`particle_count > 2³²` is supported: CUB device APIs take 64-bit `num_items` and the `idx` value
type widens to `uint64` when needed (extra scratch accounted in selection).

## Interop / Sync — unified model for both render paths

Today: PT records the reduction as a **blocking** Vulkan submit (CPU waits, reads occupied count,
builds BLAS); raster records it **inline/async** in the frame command buffer (indirect draw reads
the reduced buffer, no CPU stall). CUB unifies these:

The CUB reduction runs on a CUDA stream between the sim and the render, reading the current interop
positions (torn-latest read — same semantics as today), writing the compacted reduced positions
into an **interop buffer** the Vulkan renderer reads. The existing interop timeline semaphore
serializes CUDA→Vulkan.

- **PT path:** replace the blocking Vulkan reduction submit in `recordLodUpdate` with: launch the
  CUB reduction on the stream, `cudaStreamSynchronize` (PT already stalls here), read `num_occupied`
  from a device→host copy, then the existing AABB-writer/BLAS/trace flow consumes the reduced
  interop buffer. Nearly a straight swap.
- **Raster path:** replace the inline `recordReduction` in `recordLodRaster` with: launch the CUB
  reduction on the stream and signal the timeline; the frame's render (indirect draw) waits on the
  timeline before consuming the reduced interop buffer. The indirect-draw arg (occupied count) is
  written from the CUB `num_occupied` (a small device write or a tiny host-visible copy) instead of
  the Vulkan compaction. This is the main new plumbing (raster currently has no such wait).

The reduced-position interop buffer and the small occupied-count buffer are created as
CUDA-external (same mechanism as the position interop buffers already in `interop.cpp`).

## Determinism & Correctness

- The occupied-cell **set** is identical to the Vulkan path (a cell is occupied iff ≥1 particle
  maps to it). CUB path and Vulkan path must agree on `num_occupied`.
- Centroids: exact float `sum/count` in sorted (deterministic) cell order → reproducible run-to-run.
  (The Vulkan path uses int64 fixed-point at 2^30; CUB gives full float precision — a strict
  improvement. Cross-path comparison is within a fixed-point/float tolerance.)
- Emit order (slot assignment) is nondeterministic in both paths; the resulting reduced SET is not.
- The `--lod 32 / 2²⁰ = 1472 occupied cells` invariant must hold on **both** the CUB and Vulkan
  paths.

## Build

- Add `lib/src/lod_cub.cu` to the `mimir` library target (CUDA language already configured:
  `CUDAToolkit`, `CUDA::cudart`, `CUDA_STANDARD 20`, `CUDA_ARCHITECTURES native`). CUB is header-only
  in the CUDA toolkit (`<cub/cub.cuh>`).
- Host interface: a small C++ façade (`LodCub`), constructed with the device, particle count, grid
  N, placement; methods `reduce(stream, positions_dev, out_reduced_pos_dev, out_count)` and a
  `scratchBytes()` query for the selection step. Called from `LodContext` / the engine so the
  render paths do not depend on CUDA headers directly.

## Testing

1. **Parity:** at moderate N (e.g. 2 M, `--lod 64`), CUB `num_occupied` equals the Vulkan path's,
   and sampled centroids match within tolerance. Both placements.
2. **Invariant:** `--lod 32 / 2²⁰` headless → 1472 occupied on both paths.
3. **Fallback:** force the low-VRAM branch (env var or a tiny artificial free-VRAM cap) → Vulkan
   path selected, run still correct.
4. **>2³²:** CUB path selected on a big-VRAM GPU handles a >2³² count (or is documented B300-class).
5. **Perf:** `lod ms` on the PT stats line drops from hundreds of ms to single-digit ms on the
   B300; measured before/after.
6. Existing PT/raster LOD renders remain visually unchanged at small N.

## Risks

- The raster interop sync is new (raster currently never waits on a CUDA reduction) — the main
  integration risk; covered by parity + invariant tests.
- CUB temp-storage sizing must be queried exactly and re-checked against live free VRAM to avoid an
  OOM after the Vulkan/RT allocations; the selection step owns this.
- The B300's Vulkan-compute slowness is inferred from the sim-vs-reduction gap; the CUB path is the
  fix regardless of the precise cause (it removes the Vulkan compute from the reduction entirely).
