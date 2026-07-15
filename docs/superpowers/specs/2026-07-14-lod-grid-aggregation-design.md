# Design: `--lod N` grid-aggregation LOD for path tracing

**Date:** 2026-07-14
**Branch:** feature/remote-rendering
**Status:** Approved design, ready for implementation planning

## Problem

Path tracing in the remote-rendering sample represents every particle as one
procedural-AABB sphere in a single BLAS. At hundreds of millions of particles
(2^29 = 536,870,912 measured) two costs dominate the frame and both scale with
primitive count:

- **BLAS build/refit** ~7.2 s (refit) per frame.
- **Trace** 11.7 s at radius 0.01, and **80 s at radius 0.001** — because tiny
  spheres make the cloud semi-transparent, so rays (and their 4 bounces)
  traverse deep through the BVH hunting sparse hits instead of terminating on a
  near surface. The same transparency produces heavy Monte-Carlo noise.

Tuning the sphere radius is a dead end: bigger trades detail for a still-slow
frame; smaller is catastrophic (80 s trace + maximum noise). The only lever
that attacks build cost, trace cost, **and** the opacity/noise simultaneously is
reducing the number of BVH primitives by aggregating groups of particles into
fewer, larger, opaque stand-in spheres.

## Goal

A **reproducible benchmark knob** `--lod N`: a deterministic, fixed
(non-view-adaptive) level that trades visual fidelity for speed so we can trace
out the quality-vs-speed curve at scale. Not an automatic/adaptive system.

## Chosen approach: grid aggregation (voxel binning)

Overlay an `N x N x N` grid over the scene domain. Each occupied cell emits one
representative sphere placed at the cell's geometric center (see Cell
placement), sized to the cell. Primitive count drops from P particles to the
number of occupied cells.

The feature is a **library capability**: the aggregation passes, shaders,
buffers, and a new `ViewerOptions::pt_lod_cells` knob live in `lib/` and
`shaders/`. Samples are ordinary consumers — `rr-server.cu` only parses
`--lod N` into `options.pt_lod_cells`, exactly as it does for `--spp` /
`--bounces` / `--denoise`.

Rejected alternatives:
- **Stride decimation** (keep every k-th particle, grow radius by cbrt(k)):
  simpler, but crude (arbitrary representatives, lost fine structure). Considered
  as the easy first rung; the higher-quality grid path was chosen instead.
- **Reuse the sim's k-NN clusters** (`--k`): couples LOD to sim internals not
  exposed as clean cluster IDs. Ruled out (YAGNI).

## Key facts about the existing pipeline (grounding)

- `shaders/pathtrace_aabbs.slang`: a single compute pass writes one
  `VkAabbPositionsKHR` (24 B) per particle from the interop position buffer,
  addressed via buffer-device-address (BDA) with explicit 64-bit address
  arithmetic (NVIDIA truncates `OpPtrAccessChain` offsets to 32 bits past
  4 GiB — see that file's comment; the LOD shaders MUST follow the same
  explicit-`uint64_t`-address pattern).
- BLAS is built over those AABBs (chunked at maxPrimitiveCount ~2^29); TLAS is
  rebuilt per frame with one instance per chunk. Procedural ray-sphere
  intersection in `shaders/pathtrace.slang`.
- `shaders/pathtrace.slang` intersection shader reads `AABB[gid]` where
  `gid = InstanceID()*chunk + PrimitiveIndex()`, `chunk` packed in `cam_pos.w`.
  **No trace-shader change is required** — LOD only changes which/how-many AABBs
  exist upstream.
- Scene domain is the fixed **[-1,1]^3** cube (`rr-server.cu:354`, "Match
  datoviz/particles-kmodal-3d framing of the [-1,1]^3 domain"). Particles are
  initialized as `--k` Gaussian modes of stddev `--epsilon`; the OU walk is
  mean-reverting so particles stay bounded. This justifies a **fixed** grid
  domain — no per-frame min/max reduction pass is needed.

## Architecture

The LOD lives entirely **upstream of the BVH**. `--lod 0` (default) keeps
today's exact per-particle path untouched. `--lod N > 0` replaces only the
AABB-generation stage with three passes that produce a **compacted** list of
occupied-cell spheres in the existing AABB buffer, then builds the BLAS/TLAS
over that shorter list. BLAS/TLAS build and all trace shaders are unchanged.

### Per-frame passes (when LOD active)

Domain is the fixed padded [-1,1]^3; `cellSize = 2 / N`. Particles outside the
domain are clamped into edge cells.

1. **Clear** — `vkCmdFillBuffer` zeroes the per-cell accumulator buffer (count
   and fixed-point position sum; 0 is the identity for both). No shader.

2. **Scatter** — `shaders/pathtrace_lod_scatter.slang`, compute, dispatched over
   P particles. Each particle:
   - reads its position via explicit 64-bit BDA address arithmetic (same pattern
     as `pathtrace_aabbs.slang`);
   - computes `cell = floor((clamp(pos, -1, 1-eps) + 1) / cellSize)` per axis,
     clamped to `[0, N-1]`;
   - linear index `lin = cx + N*(cy + N*cz)`;
   - `atomicAdd(count[lin], 1u)` — count only, no position sum (the
     representative is placed at the cell CENTER, see Cell placement).

3. **Emit / compact** — `shaders/pathtrace_lod_emit.slang`, compute, dispatched
   over N^3 cells. For each cell with `count > 0`:
   - `slot = atomicAdd(globalCount, 1u)`;
   - `center = domainMin + (cellIndex + 0.5) * cellSize` (the cell's geometric
     center, derived from its 3-D index — no particle positions needed);
   - `radius = coverage * cellSize / 2` (constant `coverage` ~1.2 so neighboring
     occupied cells' spheres overlap and the aggregate stays opaque);
   - write `AABB[slot] = box(center, radius)` via explicit 64-bit BDA address.
   `globalCount` ends equal to the primitive count (number of occupied cells).

4. **Readback + build** — copy `globalCount` (4 B) to host; build the BLAS/TLAS
   over exactly that many primitives. The count read-back and the AS build are
   contained inside `recordUpdateScene`: in LOD mode it issues the clear/scatter/
   emit as an internal one-shot submit (the existing `submit()` helper), waits,
   reads the count, then records the BLAS/TLAS build into the frame command
   buffer with that exact count. No engine.cpp frame-loop change and no new
   Vulkan entry point. At multi-second frame times the extra submit+wait is
   negligible. Because `N <= 512` -> `N^3 <= 134 M < 2^29`, the result is always
   a single BLAS chunk; the existing chunking path is bypassed. LOD mode always
   does a full rebuild (no refit): the occupied-cell topology changes frame to
   frame, and the LOD primitive count is small enough that a full rebuild is
   cheap. The BLAS is sized once (in bindScene) for the worst case
   `maxCells = min(N^3, P)`; each per-frame build uses `primitiveCount = the
   read-back count <= maxCells` (allowed for a full MODE_BUILD).

## Host wiring & buffers

- **CLI:** `--lod N` parsed in `samples/remote-rendering/rr-server.cu`, mapped to
  a new `ViewerOptions::pt_lod_cells` (0 = off, the default). Reject `N > 512`
  with a clear error message (phase-1 memory cap).
- **New buffers** (allocated in `RayTracingContext::bindScene` when
  `pt_lod_cells > 0`):
  - accumulator: `N^3 * 4 B` (1 `uint` count per cell — cell-center placement
    needs no position sum). 512^3 = 537 MB.
  - `globalCount`: a 4-byte host-visible buffer for the primitive count read-back.
- **Compacted AABB output** is sized to `min(N^3, P) * 24 B` (occupied cells
  <= min(N^3, P) always). At N=512 that is 134 M * 24 B = 3.2 GB, LESS than the
  per-particle path's P * 24 B (12.9 GB at 2^29), so LOD mode reduces the AABB
  buffer allocation. When `pt_lod_cells == 0` the AABB buffer keeps its current
  `P * 24 B` size.
- **`recordUpdateScene`:** when `pt_lod_cells > 0`, replace the single
  AABB-writer dispatch with: fill -> barrier -> scatter -> barrier -> emit ->
  barrier -> readback -> build (BLAS uses the read-back count). When
  `pt_lod_cells == 0`, the current code path is unchanged.
- **New compute pipelines** for scatter and emit, created the same way as the
  existing AABB-writer pipeline (`module_path` slang compute stage, BDA push
  constants, no descriptor sets where possible).
- **Logging:** one info line per bind or per report window:
  `LOD: {N}^3 grid, {M} occupied cells from {P} particles ({P/M:.0f}:1)`.

## Determinism

This is a benchmark knob, so results must be reproducible run-to-run.

- The scatter pass uses only a `uint` count `atomicAdd` per cell. Integer
  addition is associative and order-independent, and the representative sphere's
  position is the cell's geometric center (a pure function of the cell index, no
  particle data), so the occupied-cell set and every sphere position are
  bit-identical on every run.
- The emit pass's `globalCount` atomic makes the *slot ordering* of primitives
  non-deterministic, but this affects neither the primitive **count** (=
  #occupied cells, fixed) nor the rendered **image** (the BVH is
  order-independent). Every benchmarked metric — primitive count, build ms,
  trace ms, image — is therefore reproducible.
- Cell-center placement was chosen over the mass-weighted centroid to keep this
  determinism trivial (no float/int64 atomics) and the accumulator at 4 B/cell;
  the placement difference is sub-cell at useful N (see Cell placement).
- Path-trace RNG is already frame-index-seeded and thus deterministic per frame.

## Cell placement (center vs centroid)

Each occupied cell's sphere is placed at the cell's **geometric center**
(`domainMin + (cellIndex + 0.5) * cellSize`), NOT the particle centroid. The
centroid (mass-weighted average of the cell's particle positions) tracks the
cloud silhouette marginally better in coarse/edge cells, but computing it needs
a per-cell position sum whose fixed-point form overflows a 32-bit atomic (a
single coarse cell can hold a large fraction of P ~ 5*10^8 particles), forcing
64-bit atomics (a device feature) and 32 B/cell (4.3 GB at 512^3). Cell-center
needs only a count, no extra feature, 4 B/cell, and the sphere radius already
spans the whole cell so the placement difference is sub-cell and negligible at
any N worth benchmarking.

## Radius / coverage

`cellSize = 2 / N`. Representative sphere `radius = coverage * cellSize / 2`
with `coverage` a fixed constant (default ~1.2) chosen so neighboring occupied
cells' spheres overlap and the aggregate reads as an opaque surface (removing
the transparency that caused the trace blow-up and the noise). `coverage` is an
internal tunable, not a CLI flag (YAGNI); `--lod` is the only new user-facing
knob.

## Testing & verification

No GPU unit-test harness exists in this repo, so verification is drive-the-app:

- Sweep `--lod 32 / 64 / 128 / 256` at 2^29 and confirm:
  - the logged occupied-cell count falls as expected;
  - build ms and trace ms both drop;
  - the image stays a recognizable version of the cloud (opaque blobs, not
    holes);
  - **re-running the same `--lod` yields an identical occupied-cell count**
    (determinism check).
- Confirm `--lod 0` reproduces the current per-particle behavior exactly (same
  primitive count = P, same code path).

### Edge cases

- `pt_lod_cells == 0`: untouched legacy path.
- Particles outside [-1,1]^3: clamped into edge cells.
- Empty cells: skipped (not emitted).
- All particles in one cell: 1 primitive.
- `N^3 > P`: fine (occupied cells <= P).
- `N > 512`: rejected at CLI parse with a clear message.

## Explicit non-goals (phase 1)

Documented as possible follow-ups, NOT built now:

- View-adaptive / camera-dependent resolution.
- Indirect BLAS build (`vkCmdBuildAccelerationStructuresIndirectKHR`) to remove
  the per-frame count readback — worth it only once frames are fast enough that
  the readback sync matters.
- Measured-extent (per-cell min/max) representative radii for tighter fit.
- Per-frame computed bounds instead of the fixed [-1,1]^3 domain.
- `coverage` as a CLI flag.

## Success criteria

- `--lod N` runs at 2^29 (and toward 2^30) with build ms and trace ms both
  substantially reduced versus `--lod 0`, monotonically decreasing as N falls.
- Deterministic: identical occupied-cell count across repeated runs at fixed N.
- `--lod 0` is byte-for-byte the current behavior.
- The rendered cloud remains recognizable (opaque, no holes) at usable N.
