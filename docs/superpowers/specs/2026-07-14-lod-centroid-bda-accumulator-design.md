# Design: LOD centroid placement + BDA accumulator + VRAM-scaled cap

**Date:** 2026-07-14
**Branch:** feature/remote-rendering
**Status:** Approved design, ready for implementation planning
**Builds on:** `2026-07-14-lod-grid-aggregation-design.md` (the shipped `--lod N` grid aggregation)

## Problem

The shipped `--lod N` places each occupied cell's representative sphere at the
cell's **geometric center**. Visual inspection shows two artifacts:

1. **Grid look** — spheres snap to a rigid lattice, so the aggregate reads as a
   grid of blobs rather than following the cloud's shape.
2. **Discrete jumping** — as particles drift, a cell's sphere stays pinned to the
   cell center and then pops to the next cell center; motion is not continuous.

Both come directly from cell-center placement being a pure function of the cell
index (it ignores where the particles actually are inside the cell).

Separately, the current phase-1 cap of `--lod N ≤ 512` is an arbitrary guardrail
sized for a 95 GB card. Larger-VRAM hardware (e.g. B300, 288 GB) should be able
to run finer grids.

## Goal

1. Place each occupied cell's sphere at the **mass centroid** of its particles
   (`sum(positions) / count`) instead of the cell center — so blobs follow the
   cloud and glide with particle motion, killing the grid look and reducing the
   jumping (residual pops at cell-boundary crossings and occupancy changes
   remain, but are much smaller).
2. Keep the feature a **reproducible benchmark knob**: identical result for a
   given N across runs.
3. Lift the fixed `N ≤ 512` cap to a **VRAM-scaled** limit so big-memory cards
   run finer grids, bounded by what actually fits.

## Why centroid needs a wider, integer accumulator

Centroid = `sum(positions) / count`, so the scatter pass must accumulate a
per-cell **position sum** (3 components) in addition to the count.

- The sum must be accumulated with **integer fixed-point atomics**, not float.
  Floating-point addition is not associative, so a parallel atomic float/double
  sum is order-dependent and therefore nondeterministic run-to-run — violating
  the benchmark-knob property. Integer addition is associative and exact, so a
  fixed-point integer sum is bit-identical regardless of atomic order. (This is
  the reason for integer over `float64`, not coordinate representability.)
- The sum must be **64-bit**. A coarse cell can hold a large fraction of
  P ~ 5*10^8 particles; a fixed-point sum in 32-bit atomics overflows
  (`count * scale` exceeds 2^32). `uint64` holds it: quantizing [-1,1] to a
  ~2^30 scale, a sum of ~5*10^8 values reaches ~2^60, inside 2^63. This needs
  the `shaderBufferInt64Atomics` device feature and `shaderInt64`.

Per-cell accumulator grows from **4 B** (count only) to **32 B**
(`3 * uint64` sum + `uint32` count + pad). At the grid resolutions in normal use
this is negligible (128^3 = 67 MB); it only becomes large at the top of the
range (512^3 = 4.3 GB).

## Why the accumulator must move to a BDA pointer

The accumulator is currently bound as an `RWStructuredBuffer` **descriptor**,
capped by `maxStorageBufferRange` (~4 GiB = 2^32-1 bytes on NVIDIA, regardless
of total VRAM). At 32 B/cell:

| N | cells | accumulator (32 B) | fits ~4 GiB descriptor? |
|---|-------|--------------------|-------------------------|
| 256 | 16.7 M | 537 MB | yes |
| 511 | 133 M | 4.27 GB | yes (just) |
| 512 | 134 M | 4.29 GB | **no (1 B over)** |
| 1024 | 1.07 B | 34 GB | no |

So centroid at 512^3 already exceeds the descriptor cap, and any higher N is
impossible while the accumulator is a descriptor. Moving the accumulator (count
+ sum) to a **buffer-device-address pointer** in the push constants (like the
positions and AABB buffers already are) removes the cap; the only ceiling then
is VRAM.

**Feasibility risk (must de-risk first):** this requires **64-bit integer
atomic-add through a BDA (PhysicalStorageBuffer) pointer** in Slang. Descriptor
atomics (`InterlockedAdd` on `RWStructuredBuffer`) are well-trodden; pointer
atomics are less so. The implementation plan's FIRST task is a minimal spike
that verifies int64 atomic-add through a BDA pointer compiles and produces a
correct, deterministic result on this Slang/driver/GPU combination.

### Fallback if the BDA-atomics spike fails

Keep the accumulator as a **descriptor** using `uint64` atomics
(`RWStructuredBuffer<uint64_t>` / `RWStructuredBuffer` of a struct), which caps
centroid at **N ≤ 511** (4.27 GB descriptor). This still delivers the centroid
visual fix for all practical resolutions; only the B300 high-N goal is
deferred. The plan documents this fork explicitly so the fallback is a known,
ready path rather than a redesign.

## Alternatives considered (and why centroid is the pick)

- **Representative particle (min-index) instead of centroid.** Store the
  lowest particle index that lands in each cell (`InterlockedMin` on `uint`,
  deterministic), and in emit place the sphere at that real particle's position
  (fetched via BDA). This also breaks the grid look (real positions, not the
  lattice) at only 4-8 B/cell and NO int64/BDA-atomic machinery. Rejected as the
  primary because a single representative is noisier than the mass average — when
  the min-index particle leaves the cell the representative jumps to a different
  particle, a larger pop than the centroid's smooth drift. Retained as a
  documented cheaper fallback if both the BDA and descriptor int64 paths prove
  troublesome.
- **`float64` sum.** Rejected: nondeterministic (non-associative atomics) and a
  rarer device feature than int64 atomics; buys nothing.
- **Sparse-hash accumulator (O(occupied) instead of O(N^3)).** The real escape
  from dense O(N^3) memory for extreme N. Out of scope here (larger subsystem);
  the VRAM-scaled dense cap covers the B300 target.

## Architecture

All changes stay within the existing LOD stage (scatter/emit shaders + the
accumulator buffers + bindScene sizing + one device-feature enable). The
AS-build/readback/trace machinery is unchanged. `--lod 0` remains a byte-for-byte
no-op.

### Device features (device.cpp)
- Query support for `shaderBufferInt64Atomics` (VkPhysicalDeviceVulkan12Features)
  and `shaderInt64` (VkPhysicalDeviceFeatures). Enable both when supported.
- If unsupported, LOD centroid is unavailable: fall back to cell-center
  placement for `--lod` (log a clear warning) rather than failing device
  creation. (The base cell-center path needs neither feature.)

### Accumulator (raytracing.cpp bindScene)
- Per cell: `uint32 count` + `uint64 sumX, sumY, sumZ` (fixed-point). Layout as
  two BDA buffers — the existing `lod_cellcount_buffer` (count, `uint32` * N^3)
  and a new `lod_cellsum_buffer` (`uint64` * 3 * N^3) — or one interleaved
  struct buffer; the plan picks the concrete layout. Both allocated with
  `wantAddress = true` (BDA), no longer bound as descriptors.
- The emit pass still needs the global counter (`lod_counter_buffer`,
  HOST_VISIBLE, unchanged).

### Scatter shader (pathtrace_lod_scatter.slang)
- Compute `cell` as today. `InterlockedAdd(count[lin], 1)`.
- Quantize the position to fixed-point and `InterlockedAdd` each of the 3 int64
  sum components — through BDA pointers (spike-validated) or descriptors
  (fallback). Quantization: `q = uint64((clamp(p,-1,1) + 1) * 0.5 * SCALE)`,
  `SCALE = 2^30` (documented constant).

### Emit shader (pathtrace_lod_emit.slang)
- For occupied cells: `centroid_norm = double(sum) / count / SCALE` per axis;
  `center = -1 + 2 * centroid_norm`. (Use the count already read; sums via the
  same access path as scatter.) Radius unchanged (`coverage * cellSize / 2`).

### VRAM-scaled cap (rr-server.cu + optionally a library-side check)
- Replace the hardcoded `--lod N > 512` rejection with a limit derived from the
  device-local heap size. Compute the accumulator bytes for the requested N
  (`N^3 * 32`) and reject if it exceeds a safe fraction of available
  device-local memory (query `VkPhysicalDeviceMemoryProperties` device-local
  heaps, or reuse the VRAM figures the server already gathers), reporting the
  largest N that fits. Keep a sane hard upper bound (e.g. 4096) to reject absurd
  values. Result: 95 GB -> ~1024^3 centroid; 288 GB -> higher; each card gets
  its honest ceiling.
- Emit a one-line info warning when N is in the diminishing-returns zone
  (N^3 approaching P, where occupied -> P and aggregation stops helping).

## Determinism

Preserved: `uint32` count atomics and `uint64` fixed-point sum atomics are both
integer, hence order-independent -> identical occupied-cell set and centroids
every run. The emit `globalCount` atomic still only affects slot ordering, not
count/image. Same-N reproducibility holds.

## Testing & verification

Drive-the-app (no GPU unit harness):
- **Spike (Task 1):** a standalone check that int64 atomic-add through a BDA
  pointer yields the correct, deterministic sum. Gate the rest of the plan on it.
- **Centroid correctness:** at `--lod 32` / 2^20, confirm the emitted count is
  unchanged from cell-center (aggregation topology is the same; only placement
  differs) and deterministic across two runs.
- **Visual:** at 2^29 with a client, confirm the grid look is gone and motion is
  smoother than cell-center (spheres glide, not snap).
- **High-N cap:** confirm the VRAM-scaled cap accepts an N above 512 that fits
  and rejects one that doesn't, with the reported max-N message.
- **Feature-absent path:** if `shaderBufferInt64Atomics` is unsupported (or
  simulated unsupported), `--lod` falls back to cell-center with a warning and
  still runs.
- **Regression:** `--lod 0` unchanged.

## Non-goals (this change)
- Sparse-hash accumulator (extreme N without O(N^3) memory).
- View-adaptive / hierarchical LOD.
- Eliminating all temporal popping (cell-boundary and occupancy pops remain).
- A `coverage` CLI flag.

## Success criteria
- Centroid placement visibly removes the grid look and smooths motion vs
  cell-center, at equal emitted-cell count.
- Deterministic occupied-cell count and centroids across repeated runs at fixed N.
- `--lod` runs at N > 512 on sufficient-VRAM hardware, bounded by a clear
  VRAM-scaled cap; rejects over-budget N with the max feasible N.
- `--lod 0` byte-for-byte unchanged; graceful cell-center fallback when int64
  atomics are unavailable.
