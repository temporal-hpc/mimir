# Custom-CUDA LOD Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the slow Vulkan LOD-reduction compute path with **custom native-CUDA reduction
kernels** (`size_t`-indexed, no external library) that run on the fast CUDA path (the B300 ran the
Vulkan reduction ~190× slower than CUDA) and scale past 2³², feeding both the PT and raster render
paths for both placements. The existing Vulkan reduction is retained as a fallback
(`MIMIR_LOD_NO_CUDA=1`).

**Architecture:** A standalone `LodReduce` CUDA module (clear → scatter → emit/compact kernels)
computes occupied-cell representatives from the interop positions into a CUDA-importable output
buffer. `LodContext` gains the CUDA buffers/device pointers, path selection, and a `reduceCuda`
entry; the PT and raster render paths call the CUDA reduction when active and the existing Vulkan
reduction otherwise. The existing interop timeline semaphore serializes CUDA→Vulkan.

**Tech Stack:** C++20, CUDA (custom kernels + runtime; NO CUB/Thrust), Vulkan 1.3, existing mimir
interop (`interop::Barrier`, `importCudaExternalMemory`). Spec:
`docs/superpowers/specs/2026-07-16-cub-lod-reduction-design.md` (see its REVISION section).

## Global Constraints

- **Custom CUDA kernels only** — NO CUB/Thrust (their `int` item counts cap at ~2.147 B). Index the
  particle loop with `size_t` grid-stride (like the sim), so the reduction **scales past 2³²**.
- **Placement:** cell-center = benign non-atomic occupancy store (0 atomics); centroid = `atomicAdd`
  u32 count + 3× `atomicAdd` **u64 int64 fixed-point** position sum at scale `2^30` (integer =
  order-independent = run-to-run deterministic; matches the current Vulkan shader). Emit/compact =
  one thread per N³ cell, occupied cells append via a global `atomicAdd` slot counter.
- **Selection:** the CUDA reduction is the PRIMARY path whenever CUDA is available (always for
  mimir). It holds the SAME N³ accumulator as the Vulkan path (count u32 + optional 3× u64 sum), so
  no new VRAM-fit gate is needed. `MIMIR_LOD_NO_CUDA=1` forces the Vulkan fallback. Applies to BOTH
  placements and BOTH render paths (PT, raster).
- **Determinism / parity:** the occupied-cell SET (and the occupied count) from the CUDA path must
  equal the Vulkan path's for the same input. `--lod 32` over `2^20` particles headless = **1472**
  occupied cells on BOTH paths (the invariant).
- **Centroid precision:** CUDA path uses the same int64-2^30 fixed-point as the Vulkan path (so the
  two match closely). Parity tolerance: `2 * cellSize / 2^30 + 1e-6`.
- **Counts:** `N ≤ 1625` (so `N^3 < 2^32`, cell ids are uint32) is unchanged. `particle_count` may
  exceed `2^32` — the scatter loop is `size_t`; the N³ occupancy count per cell stays u32 (a single
  cell cannot exceed the particle total, and cell counts are only used for occupancy + the centroid
  denominator, which fits u32 up to ~4.3 B/cell; if a workload could exceed that, widen the count to
  u64 — note in the report, out of scope unless triggered).
- **No behavior change when LOD is off** (`pt_lod_cells == 0`) or on the Vulkan fallback path.
- Build: library uses `-Wconversion -Werror` for C++; `lod_reduce.cu` is compiled by nvcc (CUDA lang
  already enabled on the `mimir` target). Do NOT put `-Wconversion` expectations on `.cu` files.
- Sample build for verification: `./mimir-build-from-change.sh` then
  `./samples-build-from-change.sh --sample remote-rendering`. The kmodal all-target failure on the
  pre-existing `datoviz_ext` dep is NOT a regression.

---

## File Structure

- **Create** `lib/include/private/mimir/lod_reduce.hpp` — `LodReduce` host façade (no CUDA headers
  leak; plain C++ interface so `lod.cpp`/`engine.cpp` need not be compiled by nvcc).
- **Create** `lib/src/lod_reduce.cu` — the custom scatter+emit kernels behind `LodReduce`.
- **Create** `lib/tests/lod_reduce_test.cu` — standalone parity test vs a CPU reference (built as its
  own executable target; the repo has no gtest harness).
- **Modify** `lib/include/private/mimir/lod.hpp`, `lib/src/lod.cpp` — CUDA-importable reduced-pos +
  count buffers, device-pointer accessors, selection, `reduceCuda(...)`.
- **Modify** `lib/src/raytracing.cpp` — PT path uses the CUDA reduction when active.
- **Modify** `lib/src/engine.cpp` — raster path uses the CUDA reduction when active (+ interop wait).
- **Modify** `lib/CMakeLists.txt` — add `lod_reduce.cu` to the target; add the test target.
- **Modify** `docs/…` and the `lod` stats breakout to also cover raster (Task 6).

The kernels mirror the proven Vulkan slang (`shaders/pathtrace_lod_scatter.slang` +
`pathtrace_lod_emit.slang`): same cell map, same fixed-point scale `2^30`, same benign-store /
int64-atomic split — ported to native CUDA and `size_t`-indexed.

---

### Task 1: `LodReduce` custom-CUDA reduction module (standalone + parity test)

**Files:**
- Create: `lib/include/private/mimir/lod_reduce.hpp`
- Create: `lib/src/lod_reduce.cu`
- Create: `lib/tests/lod_reduce_test.cu`
- Modify: `lib/CMakeLists.txt` (compile `lod_reduce.cu` into `mimir`; add `lod_reduce_test` executable)

**Interfaces:**
- Produces:
  ```cpp
  // lod_reduce.hpp — no CUDA headers here; opaque impl.
  namespace mimir {
  class LodReduce {
  public:
    // gridN = cells/axis; centroid = true -> mass centroid, false -> cell center.
    LodReduce(uint64_t max_particles, uint32_t gridN, bool centroid);
    ~LodReduce();
    LodReduce(const LodReduce&) = delete; LodReduce& operator=(const LodReduce&) = delete;
    // N^3 accumulator this instance holds (count u32 + optional 3*u64 sum), for logging/VRAM
    // accounting. Static so the caller can size it before constructing.
    static size_t accumulatorBytes(uint32_t gridN, bool centroid);
    // Reduce `count` positions (device ptr, packed float3, stride 12B) on `stream`:
    //  - clears the N^3 accumulator, scatters every particle (size_t grid-stride), emits one
    //    representative per occupied cell into reduced_pos_dev (compacted, float3),
    //  - returns the occupied-cell count via *occupied_dev (a single uint32 in device mem).
    // Async on `stream`; caller synchronizes/timelines before Vulkan reads reduced_pos_dev.
    void reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                void* reduced_pos_dev, uint32_t* occupied_dev);
  };
  }
  ```
- Consumes: nothing (pure CUDA).

- [ ] **Step 1: Write the failing parity test** (`lib/tests/lod_reduce_test.cu`)

Generate a deterministic host point cloud (e.g. 200k points, a few gaussian clusters), upload to
device, run `LodReduce::reduce` for a small grid (e.g. gridN=16), copy back the occupied count and
reduced positions, and compare against a CPU reference that bins the same points and computes
per-cell occupancy + centroid/center. Assert:
- occupied count matches the CPU reference exactly,
- every returned representative lies in an occupied cell and equals the CPU
  centroid/center within `2*cellSize/2^30 + 1e-6`,
- run it for both `centroid=true` and `centroid=false`,
- **determinism:** reduce the same input twice → identical occupied count and identical reduced-set
  (sort both representative lists before comparing; slot order is not deterministic, the set is).

Use the CPU reference from the retired CUB test (std::map binning, double per-cell sums, both
placements) — the reference is placement-correct and reusable verbatim.

```cpp
// lib/tests/lod_reduce_test.cu  (sketch — fill in the CPU reference + asserts)
#include "mimir/lod_reduce.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cassert>
// buildPoints(): deterministic clusters in [-1,1]^3. cpuReduce(): reference occupancy+centroid.
// For each placement: upload, reduce, download, compare (set-equality); reduce twice, compare.
int main() { /* ... deterministic points, CPU ref, compare, print PASS/FAIL ... */ }
```

- [ ] **Step 2: Add the test target and run it to see it fail** (compile error / missing symbols)

In `lib/CMakeLists.txt` add (guarded so it does not affect the default build if tests are off):
```cmake
add_executable(lod_reduce_test tests/lod_reduce_test.cu src/lod_reduce.cu)
target_include_directories(lod_reduce_test PRIVATE include/private include/public)
target_link_libraries(lod_reduce_test PRIVATE CUDA::cudart)
set_target_properties(lod_reduce_test PROPERTIES CUDA_STANDARD 20 CUDA_ARCHITECTURES native)
```
Run: `cmake --build build --target lod_reduce_test` — expected FAIL (no `LodReduce` impl yet).

- [ ] **Step 3: Implement `LodReduce`** (`lod_reduce.hpp` + `lod_reduce.cu`)

`lod_reduce.cu` holds a private `Impl` (pImpl) so CUDA types stay out of the public header. Three
custom kernels — NO CUB/Thrust — mirroring the slang scatter+emit exactly:

```cuda
static __device__ __forceinline__ uint32_t cellId(float px, float py, float pz, uint32_t N) {
  float n = float(N);
  int cx = int(fminf(fmaxf((px + 1.f) * 0.5f * n, 0.f), n - 1.f));
  int cy = int(fminf(fmaxf((py + 1.f) * 0.5f * n, 0.f), n - 1.f));
  int cz = int(fminf(fmaxf((pz + 1.f) * 0.5f * n, 0.f), n - 1.f));
  return uint32_t(cx) + N * (uint32_t(cy) + N * uint32_t(cz)); // N<=1625 => N^3 < 2^32
}
// clear: zero counts[N^3] (and sums[3*N^3] when centroid).
__global__ void clearKernel(uint32_t* counts, unsigned long long* sums, uint32_t nCells, bool centroid);
// scatter: size_t grid-stride over `count` particles (like the sim).
//  centroid : atomicAdd(&counts[c],1) + 3x atomicAdd(&sums[3c+k], fixedpoint(pos_k))  (u64, scale 2^30)
//  cell-cent: counts[c] = 1u  (benign non-atomic store; every writer writes 1)
__global__ void scatterKernel(const float* pos, uint64_t count, uint32_t N, bool centroid,
                              uint32_t* counts, unsigned long long* sums);
// emit: one thread per N^3 cell; occupied (counts[c]!=0) append via atomicAdd(occupied,1) slot.
//  cell-cent: center = -1 + (cell + 0.5) * (2/N) per axis.
//  centroid : de-fixedpoint sums[3c+k]/counts[c] back to [-1,1].
__global__ void emitKernel(const uint32_t* counts, const unsigned long long* sums, uint32_t nCells,
                           uint32_t N, bool centroid, float* reduced_pos, uint32_t* occupied);
```

Fixed-point matches the slang: `q = (unsigned long long)((clamp(p,-1,1)+1)*0.5*2^30)`; de-quantize
`p = (double(sum)/count) / 2^30 * 2 - 1`. `reduce(...)` on `stream`: `cudaMemsetAsync(occupied,0)`,
`clearKernel`, `scatterKernel` (grid sized to saturate the device; the `size_t` loop covers any
`count`), `emitKernel`. Allocate `counts[N^3]` (u32) and, when centroid, `sums[3*N^3]` (u64) once in
the ctor; free in dtor. `accumulatorBytes = N^3*4 + (centroid ? N^3*24 : 0)`. **Index the scatter
loop with `size_t`/`uint64_t` — never `int`** (this is the whole point: scale past 2³²). Check every
launch with `cudaGetLastError()` after the kernel and `cudaPeekAtLastError()`; surface failures.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cmake --build build --target lod_reduce_test && ./build/lod_reduce_test`
Expected: `PASS` for both placements (occupied count + representatives match the CPU reference; the
twice-reduce determinism check matches).

- [ ] **Step 5: Commit**
```bash
git add lib/include/private/mimir/lod_reduce.hpp lib/src/lod_reduce.cu lib/tests/lod_reduce_test.cu lib/CMakeLists.txt
git commit -m "feat(lod): LodReduce custom-CUDA scatter+emit reduction module + parity test"
```

---

### Task 2: CUDA-importable reduced-position + count buffers in `LodContext`

**Files:**
- Modify: `lib/include/private/mimir/lod.hpp`, `lib/src/lod.cpp`

**Interfaces:**
- Consumes: `interop::importCudaExternalMemory(VkDeviceMemory, VkDeviceSize, VkDevice)` and
  `cudaExternalMemoryGetMappedBuffer` (follow the existing interop buffer pattern in
  `lib/src/interop.cpp` / how position interop buffers are created in `interop.cpp`/`engine.cpp`).
- Produces:
  ```cpp
  // lod.hpp additions
  void*    reducedPositionsDevicePtr(uint32_t slot) const; // CUDA ptr aliasing reduced_pos[slot]
  uint32_t* occupiedDevicePtr(uint32_t slot) const;        // CUDA ptr to a device uint32 count
  ```

- [ ] **Step 1: Read the existing interop buffer creation** in `lib/src/interop.cpp` and where the
  engine imports the CUDA position buffer, to mirror the exportable-memory + `cudaExternalMemory`
  import pattern exactly (VkExportMemoryAllocateInfo on the buffer's memory, then
  `cudaExternalMemoryGetMappedBuffer`).

- [ ] **Step 2: Make `reduced_pos[slot]` exportable + import to CUDA.** In the LodContext buffer
  allocation (where `reduced_pos[slot]` is created), add the external-memory export handle type, and
  after allocation import a CUDA device pointer for each slot. Add a small per-slot device `uint32`
  occupied-count buffer (exportable, imported to CUDA) OR reuse the existing HOST_VISIBLE counter's
  device pointer — pick the one that lets both the CUDA kernel write and `readCount` read. Store the
  CUDA ptrs.

- [ ] **Step 3: Add the accessors** `reducedPositionsDevicePtr(slot)` / `occupiedDevicePtr(slot)`.

- [ ] **Step 4: Guard: only allocate/import the CUDA aliases when the CUDA path is active**
  (Task 3 sets the flag; for this task, gate on a `use_cuda` member defaulting false, wired in Task
  3). Build the library; confirm no regression to the Vulkan path (LOD still inits, `1472` invariant
  unchanged since the CUDA path is not yet wired).

- [ ] **Step 5: Verify + commit**
Run `./mimir-build-from-change.sh` (clean) and a `--lod 32` `2^20` headless PT run → still 1472.
```bash
git add lib/include/private/mimir/lod.hpp lib/src/lod.cpp
git commit -m "feat(lod): CUDA-importable reduced-position + occupied-count buffers"
```

---

### Task 3: Path selection (CUDA primary, Vulkan fallback) + `reduceCuda` entry in `LodContext`

**Files:**
- Modify: `lib/include/private/mimir/lod.hpp`, `lib/src/lod.cpp`

**Interfaces:**
- Consumes: `LodReduce` (Task 1), the CUDA device ptrs (Task 2), the interop positions CUDA device
  pointer + the interop `cudaStream_t` (from `interop::Barrier::cuda_stream`; the engine passes it
  in — see Task 4/5).
- Produces:
  ```cpp
  bool usesCuda() const;  // true when the CUDA reduction path is active
  // Run the CUDA reduction for `slot`: reads positions_dev, writes reduced_pos + occupied for `slot`.
  // No-op / must-not-be-called when !usesCuda().
  void reduceCuda(cudaStream_t stream, const void* positions_dev, uint64_t count, uint32_t slot);
  uint32_t occupiedFromCuda(uint32_t slot); // device->host copy of the CUDA occupied count for `slot`
  ```

- [ ] **Step 1:** In `LodContext::init`, after computing `grid_n`/`max_cells`, set
  `use_cuda = (getenv("MIMIR_LOD_NO_CUDA") == nullptr)`. No VRAM-fit gate — the CUDA path holds the
  SAME N³ accumulator as the Vulkan path (`LodReduce::accumulatorBytes(grid_n, centroid_active)`), so
  if LOD was allowed at all it fits. When `use_cuda`, construct the `LodReduce` instance and (Task 2)
  import the CUDA buffer aliases. Log the choice: `"LOD reduction: custom CUDA kernels"` or
  `"LOD reduction: Vulkan scatter (MIMIR_LOD_NO_CUDA set)"`.
- [ ] **Step 2:** Implement `reduceCuda` (calls `LodReduce::reduce(stream, positions_dev, count,
  reducedPositionsDevicePtr(slot), occupiedDevicePtr(slot))`) and `occupiedFromCuda` (cudaMemcpy the
  device count to host, clamp to `maxCells()`).
- [ ] **Step 3:** Build the library. Confirm `usesCuda()` is true by default and false under
  `MIMIR_LOD_NO_CUDA=1`; confirm the Vulkan path still selected+correct when `use_cuda` false. (The
  end-to-end reduce is exercised in Tasks 4/5; here just verify construction/selection + a clean
  `./mimir-build-from-change.sh`.)
- [ ] **Step 4: Commit**
```bash
git add lib/include/private/mimir/lod.hpp lib/src/lod.cpp
git commit -m "feat(lod): CUDA-primary path selection + reduceCuda entry (MIMIR_LOD_NO_CUDA fallback)"
```

---

### Task 4: Path-tracing integration

**Files:**
- Modify: `lib/src/raytracing.cpp` (`recordLodUpdate`), and its caller in `lib/src/engine.cpp` if it
  must pass the interop stream/positions device ptr.

**Interfaces:**
- Consumes: `LodContext::usesCuda()`, `reduceCuda(...)`, `occupiedFromCuda(...)`; the interop
  `cudaStream_t` and the positions CUDA device pointer (thread through from the engine — the engine
  already owns the interop barrier/stream and the CUDA position buffer).

- [ ] **Step 1:** In `recordLodUpdate`, branch on `lod->usesCuda()`:
  - **CUDA:** `lod->reduceCuda(stream, positions_dev, particle_count, /*slot=*/0)`, then
    `cudaStreamSynchronize(stream)` (PT already stalls here), `occupied = lod->occupiedFromCuda(0)`.
    Skip the Vulkan reduction submit. The subsequent AABB writer reads
    `lod->reducedPositionsAddress(0)` exactly as today (the buffer now holds the CUDA output).
  - **Vulkan (else):** unchanged (`submit([&]{ lod->recordReduction(...) })` + `readCount`), with the
    existing `last_lod_ms` CPU timing.
  Also CPU-time the CUDA branch into `last_lod_ms` (wrap the `reduceCuda`+sync).
- [ ] **Step 2:** Thread the interop `cuda_stream` and positions device ptr into `recordLodUpdate`
  (add params or a small accessor on the engine/rt context). Keep the RtBuffer/BDA address flow for
  the AABB writer unchanged.
- [ ] **Step 3: Verify** on the local GPU: PT `--lod 256` at 2M and at ~300M, CUDA path active:
  - the render is visually non-blank (headless client 5 frames, ppm non-empty),
  - `LOD emitted N occupied cells` and the PT `lod X ms` line appear; `lod` is far lower than the
    Vulkan path at the same N,
  - `--lod 32` `2^20` still 1472 occupied.
- [ ] **Step 4: Commit**
```bash
git add lib/src/raytracing.cpp lib/src/engine.cpp
git commit -m "feat(lod): PT path uses the custom CUDA reduction when active"
```

---

### Task 5: Raster path integration (interop sync — the new plumbing)

**Files:**
- Modify: `lib/src/engine.cpp` (`recordLodRaster`, and the frame submit/wait in `renderFrame`)

**Interfaces:**
- Consumes: `LodContext::usesCuda()`, `reduceCuda`, `occupiedFromCuda`, `indirectBuffer(slot)`, the
  interop `Barrier` (timeline semaphore) already owned by the engine.

- [ ] **Step 1: Read** how `renderFrame` currently submits the frame command buffer and how the
  interop timeline semaphore is waited/signaled for the sim (`render_timeline`, `interop::Barrier`),
  so the CUDA reduction's signal is inserted correctly (CUDA signals the timeline after `reduceCuda`;
  the frame's queue submit adds a wait on that timeline value before the draw).

- [ ] **Step 2:** In `recordLodRaster`, branch on `usesCuda()`:
  - **CUDA:** do NOT record the inline Vulkan `recordReduction`. Instead (recorded/scheduled around
    the frame): run `lod->reduceCuda(stream, positions_dev, lod_raster_count, slot)` on the interop
    stream and signal the timeline; the engine adds a wait on that timeline value to the frame's
    graphics submit so the indirect draw reads a completed `reducedPositionsBuffer(slot)`. Write the
    occupied count into the indirect-args buffer from the CUDA count — either a `cudaMemcpy` into a
    CUDA alias of the indirect buffer's varying field, or keep `recordIndirectArgs` but feed it the
    CUDA count (device value) instead of the Vulkan counter. Choose the smaller change; document it.
  - **Vulkan (else):** unchanged (`recordReduction` inline + `recordIndirectArgs`).
- [ ] **Step 3:** Ensure the reduced-position VERTEX buffer barrier/visibility is correct across the
  CUDA→Vulkan boundary (the timeline wait provides execution+memory availability; add the
  VERTEX_ATTRIBUTE_READ barrier if the existing one is inside the removed inline path).
- [ ] **Step 4: Verify** on the local GPU: `--light-model none --lod 256` and `phong --lod 256` at
  ~300M with the CUDA path active:
  - render non-blank, occupied count matches the Vulkan path at the same N,
  - `--lod 32` `2^20` none = 1472 occupied,
  - no validation errors (run with validation if available).
- [ ] **Step 5: Commit**
```bash
git add lib/src/engine.cpp
git commit -m "feat(lod): raster path uses the custom CUDA reduction + interop wait when active"
```

---

### Task 6: Parity/fallback/perf verification, raster `lod` timing, docs

**Files:**
- Modify: `lib/src/engine.cpp` (raster `lod` timing breakout), `lib/src/remote.cpp` (surface it),
  `samples/remote-rendering/rr-server.cu` (usage note), the spec/plan status.

- [ ] **Step 1: Raster `lod` timing.** Add a CPU (CUDA path) / GPU-timestamp (Vulkan path) measure of
  the raster reduction and surface `lod X ms` on the raster `[stats]` line, mirroring the PT
  breakout, so none/phong show their reduction cost too.
- [ ] **Step 2: Parity gate.** Headless: for PT and raster, at `--lod 64` over 2M points, assert the
  CUDA occupied count equals a `MIMIR_LOD_NO_CUDA=1` (Vulkan-path) run's occupied count (same
  seed/k/epsilon). Record both in the task report.
- [ ] **Step 3: Fallback.** Run with `MIMIR_LOD_NO_CUDA=1` and confirm a correct render + the log says
  `Vulkan scatter (MIMIR_LOD_NO_CUDA set)`. This is the documented escape hatch (already wired in
  Task 3); just verify it end-to-end here.
- [ ] **Step 4: Perf.** Record `lod ms` CUDA vs Vulkan (`MIMIR_LOD_NO_CUDA=1`) at 300M (and 1B if
  VRAM allows) for PT and raster; expect single-digit ms on CUDA. Put the numbers in the task report.
  **If centroid `lod ms` on the B300 is still large (atomic contention), record it — that is the
  trigger for the follow-up contention-free sort noted in the spec.**
- [ ] **Step 5: Docs.** Note the custom CUDA reduction + `MIMIR_LOD_NO_CUDA` in `rr-server
  --help`/README and the `pt_lod_cells`/`lod_centroid` option docs; mark the plan complete.
- [ ] **Step 6: Commit**
```bash
git add -A
git commit -m "feat(lod): raster lod timing + CUDA parity/fallback/perf verification + docs"
```

---

## Self-Review

- **Spec coverage:** selection (T3), custom CUDA kernels (T1), interop buffers (T2), PT integration
  (T4), raster integration (T5), determinism/parity + fallback + perf + docs (T6). All spec sections
  map to a task.
- **Placeholders:** the kernel sketch gives concrete CUDA (cell map, fixed-point, benign store vs
  int64 atomics, emit slot counter); integration tasks give exact functions/anchors and require
  reading the named existing patterns (interop.cpp, renderFrame) — no "TODO/handle later". The
  literal interop-export and frame-wait code is specified as "mirror the existing pattern at <file>"
  rather than invented, because it depends on internals the implementer must read to match exactly.
- **Type consistency:** `LodReduce::reduce` signature, `reduceCuda`/`occupiedFromCuda`/`usesCuda`
  names, and `reducedPositionsDevicePtr`/`occupiedDevicePtr` accessors are used consistently across
  T1–T5. `MIMIR_LOD_NO_CUDA` is the single fallback env var (T3 wires it, T6 verifies it).
- **Scale:** the scatter loop is `size_t`-indexed everywhere (T1 kernel + constraint), so no
  `INT32_MAX`/`int` cap; `particle_count > 2³²` is supported (unlike the retired CUB approach).
- **Risk ordering:** T1 (pure CUDA, unit-tested) → T2/T3 (buffers/selection, no path change) → T4
  (PT, near-swap) → T5 (raster, the risky interop sync) → T6 (verify/docs). Each task is
  independently testable and leaves the tree building with the Vulkan path intact until its path is
  switched.
