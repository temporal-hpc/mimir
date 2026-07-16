# CUB-Based LOD Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native-CUDA (CUB) LOD-reduction fast path that bypasses the slow Vulkan compute
path on datacenter GPUs and beats the atomic scatter elsewhere, runtime-selected when its scratch
fits free VRAM, with the existing Vulkan reduction as universal fallback. Covers both PT and raster
paths and both placements.

**Architecture:** A standalone `LodCub` CUDA module (sort → run-length-encode → reduce-by-key →
finalize) computes occupied-cell representatives from the interop positions into a CUDA-importable
output buffer. `LodContext` gains the buffers, device pointers, selection, and a
`recordReductionCuda`-style entry; the PT and raster render paths call CUB when selected and the
existing Vulkan reduction otherwise. The existing interop timeline semaphore serializes CUDA→Vulkan.

**Tech Stack:** C++20, CUDA/CUB (header-only, in the toolkit), Vulkan 1.3, existing mimir interop
(`interop::Barrier`, `importCudaExternalMemory`). Spec:
`docs/superpowers/specs/2026-07-16-cub-lod-reduction-design.md`.

## Global Constraints

- **Selection:** CUB fast path is used when its total scratch (sort keys/values 2× each + sorted
  outputs + RLE/reduce outputs + `cub_temp`, all queried exactly) fits currently-free VRAM
  (`cudaMemGetInfo`) with margin; otherwise the existing Vulkan reduction runs. Applies to BOTH
  placements (centroid and cell-center) and BOTH render paths (PT, raster).
- **Determinism / parity:** the occupied-cell SET (and thus the occupied count) from the CUB path
  must equal the Vulkan path's for the same input. `--lod 32` over `2^20` particles headless =
  **1472** occupied cells on BOTH paths (the invariant).
- **Centroid precision:** CUB centroids are exact float `sum/count` in sorted cell order
  (deterministic); this replaces the Vulkan int64-2^30-fixed-point centroid. Cross-path centroid
  comparison uses a tolerance of `2 * cellSize / 2^30 + 1e-6`.
- **Counts:** `N ≤ 1625` (so `N^3 < 2^32`, cell ids are uint32) is unchanged. `particle_count` may
  exceed `2^32` (CUB `num_items` is 64-bit; the value/index type widens to uint64 when
  `count > 2^32`).
- **No behavior change when LOD is off** (`pt_lod_cells == 0`) or on the Vulkan fallback path.
- Build: library uses `-Wconversion -Werror` for C++; `lod_cub.cu` is compiled by nvcc (CUDA lang
  already enabled on the `mimir` target). Do NOT put `-Wconversion` expectations on `.cu` files.
- Sample build for verification: `./mimir-build-from-change.sh` then
  `./samples-build-from-change.sh --sample remote-rendering`. The kmodal all-target failure on the
  pre-existing `datoviz_ext` dep is NOT a regression.

---

## File Structure

- **Create** `lib/include/private/mimir/lod_cub.hpp` — `LodCub` host façade (no CUB headers leak;
  plain C++ interface so `lod.cpp`/`engine.cpp` need not be compiled by nvcc).
- **Create** `lib/src/lod_cub.cu` — the CUB pipeline + kernels behind `LodCub`.
- **Create** `lib/tests/lod_cub_test.cu` — standalone parity test vs a CPU reference (built as its
  own executable target; the repo has no gtest harness).
- **Modify** `lib/include/private/mimir/lod.hpp`, `lib/src/lod.cpp` — CUB-importable reduced-pos +
  count buffers, device-pointer accessors, selection, `reduceCuda(...)`.
- **Modify** `lib/src/raytracing.cpp` — PT path uses CUB when selected.
- **Modify** `lib/src/engine.cpp` — raster path uses CUB when selected (+ the interop wait).
- **Modify** `lib/CMakeLists.txt` — add `lod_cub.cu` to the target; add the test target.
- **Modify** `docs/…` and the `lod` stats breakout to also cover raster (Task 6).

---

### Task 1: `LodCub` CUDA/CUB reduction module (standalone + parity test)

**Files:**
- Create: `lib/include/private/mimir/lod_cub.hpp`
- Create: `lib/src/lod_cub.cu`
- Create: `lib/tests/lod_cub_test.cu`
- Modify: `lib/CMakeLists.txt` (compile `lod_cub.cu` into `mimir`; add `lod_cub_test` executable)

**Interfaces:**
- Produces:
  ```cpp
  // lod_cub.hpp — no CUDA/CUB headers here; opaque impl.
  namespace mimir {
  class LodCub {
  public:
    // gridN = cells/axis; centroid = true -> mass centroid, false -> cell center.
    LodCub(uint64_t max_particles, uint32_t gridN, bool centroid);
    ~LodCub();
    LodCub(const LodCub&) = delete; LodCub& operator=(const LodCub&) = delete;
    // Total device scratch this instance holds (for the VRAM-fit selection). Static so the caller
    // can decide BEFORE constructing.
    static size_t scratchBytes(uint64_t max_particles, uint32_t gridN, bool centroid);
    // Reduce `count` positions (device ptr, packed float3, stride 12B) on `stream`:
    //  - writes the compacted representative positions (float3) to reduced_pos_dev,
    //  - returns the occupied-cell count via *occupied_dev (a single uint32 in device mem).
    // Async on `stream`; caller synchronizes/timelines before Vulkan reads reduced_pos_dev.
    void reduce(cudaStream_t stream, const void* positions_dev, uint64_t count,
                void* reduced_pos_dev, uint32_t* occupied_dev);
  };
  }
  ```
- Consumes: nothing (pure CUDA).

- [ ] **Step 1: Write the failing parity test** (`lib/tests/lod_cub_test.cu`)

Generate a deterministic host point cloud (e.g. 200k points, a few gaussian clusters), upload to
device, run `LodCub::reduce` for a small grid (e.g. gridN=16), copy back the occupied count and
reduced positions, and compare against a CPU reference that bins the same points and computes
per-cell occupancy + centroid/center. Assert:
- occupied count matches the CPU reference exactly,
- every returned representative lies in an occupied cell and equals the CPU
  centroid/center within `2*cellSize/2^30 + 1e-6`,
- run it for both `centroid=true` and `centroid=false`.

```cpp
// lib/tests/lod_cub_test.cu  (sketch — fill in the CPU reference + asserts)
#include "mimir/lod_cub.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cassert>
// buildPoints(): deterministic clusters in [-1,1]^3. cpuReduce(): reference occupancy+centroid.
// For each placement: upload, reduce, download, compare. Return non-zero on mismatch.
int main() { /* ... deterministic points, CPU ref, compare, print PASS/FAIL ... */ }
```

- [ ] **Step 2: Add the test target and run it to see it fail** (compile error / missing symbols)

In `lib/CMakeLists.txt` add (guarded so it does not affect the default build if tests are off):
```cmake
add_executable(lod_cub_test tests/lod_cub_test.cu src/lod_cub.cu)
target_include_directories(lod_cub_test PRIVATE include/private include/public)
target_link_libraries(lod_cub_test PRIVATE CUDA::cudart)
set_target_properties(lod_cub_test PROPERTIES CUDA_STANDARD 20 CUDA_ARCHITECTURES native)
```
Run: `cmake --build build --target lod_cub_test` — expected FAIL (no `LodCub` impl yet).

- [ ] **Step 3: Implement `LodCub`** (`lod_cub.hpp` + `lod_cub.cu`)

`lod_cub.cu` holds a private `Impl` (pImpl) so CUB headers stay out of the public header. Pipeline:
```cuda
// cell id: clamp pos to [-1,1], map to [0,N), linearize -> uint32 key; value = index.
__global__ void keyKernel(const float* pos, uint64_t n, uint32_t N, uint32_t* keys, uint32_t* idx);
// finalize: one thread per occupied unique cell -> reduced_pos[k].
//  cell-center: delinearize unique_key -> geometric center.
//  centroid:    sum[k] / count[k].
__global__ void finalizeKernel(const uint32_t* uniq, const uint32_t* counts,
                               const float3* sums, uint32_t occupied, uint32_t N,
                               bool centroid, float* reduced_pos);
```
Steps in `reduce(...)` (all on `stream`, buffers pre-allocated in ctor):
1. `keyKernel` → keys, idx.
2. `cub::DeviceRadixSort::SortPairs(temp, keys, keys_out, idx, idx_out, n, 0, bitsForN)` where
   `bitsForN = ceil(log2(N*N*N))`.
3. `cub::DeviceRunLengthEncode::Encode(temp, keys_out, uniq, counts, num_runs_dev, n)` → occupied
   set + counts + `*num_runs_dev` (the occupied count). `cudaMemcpy` it to `occupied_dev`.
4. centroid only: gather positions by `idx_out` (a `cub::TransformInputIterator` or a small gather
   kernel producing float3 in sorted order) and `cub::DeviceReduceByKey(temp, keys_out,
   gathered_pos, dummy_keys_out, sums, num_runs2_dev, Float3Add(), n)` → per-unique-cell sums
   aligned with `uniq`.
5. `finalizeKernel` → `reduced_pos_dev`.
Allocate `keys, keys_out, idx, idx_out (2*n each), uniq, counts, sums (<= min(N^3, n)),
num_runs_dev, cub_temp` in the ctor (cudaMalloc), free in dtor. `scratchBytes` sums those sizes
(query `cub_temp` bytes with a null-temp pass at ctor-time sizes). Widen `idx`/value to `uint64`
when `max_particles > UINT32_MAX`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cmake --build build --target lod_cub_test && ./build/lod_cub_test`
Expected: `PASS` for both placements (occupied count + representatives match the CPU reference).

- [ ] **Step 5: Commit**
```bash
git add lib/include/private/mimir/lod_cub.hpp lib/src/lod_cub.cu lib/tests/lod_cub_test.cu lib/CMakeLists.txt
git commit -m "feat(lod): LodCub CUDA/CUB reduction module + parity test"
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
  device pointer — pick the one that lets both CUB write and `readCount` read. Store the CUDA ptrs.

- [ ] **Step 3: Add the accessors** `reducedPositionsDevicePtr(slot)` / `occupiedDevicePtr(slot)`.

- [ ] **Step 4: Guard: only allocate/import the CUDA aliases when the CUB path is selected**
  (Task 3 sets the flag; for this task, gate on a `use_cub` member defaulting false, wired in Task
  3). Build the library; confirm no regression to the Vulkan path (LOD still inits, `1472` invariant
  unchanged since CUB is not yet selected).

- [ ] **Step 5: Verify + commit**
Run `./mimir-build-from-change.sh` (clean) and a `--lod 32` `2^20` headless PT run → still 1472.
```bash
git add lib/include/private/mimir/lod.hpp lib/src/lod.cpp
git commit -m "feat(lod): CUDA-importable reduced-position + occupied-count buffers"
```

---

### Task 3: Selection (CUB-when-it-fits) + `reduceCuda` entry in `LodContext`

**Files:**
- Modify: `lib/include/private/mimir/lod.hpp`, `lib/src/lod.cpp`

**Interfaces:**
- Consumes: `LodCub` (Task 1), the CUDA device ptrs (Task 2), the interop positions CUDA device
  pointer + the interop `cudaStream_t` (from `interop::Barrier::cuda_stream`; the engine passes it
  in — see Task 4/5).
- Produces:
  ```cpp
  bool usesCub() const;   // true when the CUB fast path was selected at init
  // Run the CUB reduction for `slot`: reads positions_dev, writes reduced_pos + occupied for `slot`.
  // No-op / must-not-be-called when !usesCub().
  void reduceCuda(cudaStream_t stream, const void* positions_dev, uint64_t count, uint32_t slot);
  uint32_t occupiedFromCuda(uint32_t slot); // device->host copy of the CUB occupied count for `slot`
  ```

- [ ] **Step 1:** In `LodContext::init`, after computing `grid_n`/`max_cells`, compute
  `LodCub::scratchBytes(particle_count, grid_n, centroid_active)`, query `cudaMemGetInfo`, and set
  `use_cub = (scratch + reduced/count buffers) + margin <= free`. When `use_cub`, construct the
  `LodCub` instance and (Task 2) import the CUDA buffer aliases. Log the choice
  (`"LOD reduction: CUB (CUDA) fast path"` / `"... Vulkan scatter (CUB scratch N GB > free)"`).
- [ ] **Step 2:** Implement `reduceCuda` (calls `LodCub::reduce(stream, positions_dev, count,
  reducedPositionsDevicePtr(slot), occupiedDevicePtr(slot))`) and `occupiedFromCuda` (cudaMemcpy the
  device count to host, clamp to `maxCells()`).
- [ ] **Step 3:** Build; add a tiny standalone check or reuse Task 1's test to confirm selection math
  (e.g. a unit assert that `scratchBytes` is > 0 and selection returns true on a large free value,
  false on a tiny one). Confirm the Vulkan path still selected+correct when `use_cub` false.
- [ ] **Step 4: Commit**
```bash
git add lib/include/private/mimir/lod.hpp lib/src/lod.cpp
git commit -m "feat(lod): runtime CUB-when-it-fits selection + reduceCuda entry"
```

---

### Task 4: Path-tracing integration

**Files:**
- Modify: `lib/src/raytracing.cpp` (`recordLodUpdate`), and its caller in `lib/src/engine.cpp` if it
  must pass the interop stream/positions device ptr.

**Interfaces:**
- Consumes: `LodContext::usesCub()`, `reduceCuda(...)`, `occupiedFromCuda(...)`; the interop
  `cudaStream_t` and the positions CUDA device pointer (thread through from the engine — the engine
  already owns the interop barrier/stream and the CUDA position buffer).

- [ ] **Step 1:** In `recordLodUpdate`, branch on `lod->usesCub()`:
  - **CUB:** `lod->reduceCuda(stream, positions_dev, particle_count, /*slot=*/0)`, then
    `cudaStreamSynchronize(stream)` (PT already stalls here), `occupied = lod->occupiedFromCuda(0)`.
    Skip the Vulkan reduction submit. The subsequent AABB writer reads
    `lod->reducedPositionsAddress(0)` exactly as today (the buffer now holds the CUB output).
  - **Vulkan (else):** unchanged (`submit([&]{ lod->recordReduction(...) })` + `readCount`), with the
    existing `last_lod_ms` CPU timing.
  Also CPU-time the CUB branch into `last_lod_ms` (wrap the `reduceCuda`+sync).
- [ ] **Step 2:** Thread the interop `cuda_stream` and positions device ptr into `recordLodUpdate`
  (add params or a small accessor on the engine/rt context). Keep the RtBuffer/BDA address flow for
  the AABB writer unchanged.
- [ ] **Step 3: Verify** on the local GPU: PT `--lod 256` at 2M and at ~300M, CUB selected:
  - the render is visually non-blank (headless client 5 frames, ppm non-empty),
  - `LOD emitted N occupied cells` and the PT `lod X ms` line appear; `lod` is far lower than the
    Vulkan path at the same N,
  - `--lod 32` `2^20` still 1472 occupied.
- [ ] **Step 4: Commit**
```bash
git add lib/src/raytracing.cpp lib/src/engine.cpp
git commit -m "feat(lod): PT path uses the CUB reduction when selected"
```

---

### Task 5: Raster path integration (interop sync — the new plumbing)

**Files:**
- Modify: `lib/src/engine.cpp` (`recordLodRaster`, and the frame submit/wait in `renderFrame`)

**Interfaces:**
- Consumes: `LodContext::usesCub()`, `reduceCuda`, `occupiedFromCuda`, `indirectBuffer(slot)`, the
  interop `Barrier` (timeline semaphore) already owned by the engine.

- [ ] **Step 1: Read** how `renderFrame` currently submits the frame command buffer and how the
  interop timeline semaphore is waited/signaled for the sim (`render_timeline`, `interop::Barrier`),
  so the CUB reduction's signal is inserted correctly (CUDA signals the timeline after `reduceCuda`;
  the frame's queue submit adds a wait on that timeline value before the draw).

- [ ] **Step 2:** In `recordLodRaster`, branch on `usesCub()`:
  - **CUB:** do NOT record the inline Vulkan `recordReduction`. Instead (recorded/scheduled around
    the frame): run `lod->reduceCuda(stream, positions_dev, lod_raster_count, slot)` on the interop
    stream and signal the timeline; the engine adds a wait on that timeline value to the frame's
    graphics submit so the indirect draw reads a completed `reducedPositionsBuffer(slot)`. Write the
    occupied count into the indirect-args buffer from the CUB count — either a `cudaMemcpy` into a
    CUDA alias of the indirect buffer's varying field, or keep `recordIndirectArgs` but feed it the
    CUB count (device value) instead of the Vulkan counter. Choose the smaller change; document it.
  - **Vulkan (else):** unchanged (`recordReduction` inline + `recordIndirectArgs`).
- [ ] **Step 3:** Ensure the reduced-position VERTEX buffer barrier/visibility is correct across the
  CUDA→Vulkan boundary (the timeline wait provides execution+memory availability; add the
  VERTEX_ATTRIBUTE_READ barrier if the existing one is inside the removed inline path).
- [ ] **Step 4: Verify** on the local GPU: `--light-model none --lod 256` and `phong --lod 256` at
  ~300M with CUB selected:
  - render non-blank, occupied count matches the Vulkan path at the same N,
  - `--lod 32` `2^20` none = 1472 occupied,
  - no validation errors (run with validation if available).
- [ ] **Step 5: Commit**
```bash
git add lib/src/engine.cpp
git commit -m "feat(lod): raster path uses the CUB reduction + interop wait when selected"
```

---

### Task 6: Parity/fallback/perf verification, raster `lod` timing, docs

**Files:**
- Modify: `lib/src/engine.cpp` (raster `lod` timing breakout), `lib/src/remote.cpp` (surface it),
  `samples/remote-rendering/rr-server.cu` (usage note), the spec/plan status.

- [ ] **Step 1: Raster `lod` timing.** Add a CPU (CUB path) / GPU-timestamp (Vulkan path) measure of
  the raster reduction and surface `lod X ms` on the raster `[stats]` line, mirroring the PT
  breakout, so none/phong show their reduction cost too.
- [ ] **Step 2: Parity gate.** Headless: for PT and raster, at `--lod 64` over 2M points, assert the
  CUB occupied count equals a Vulkan-path run's occupied count (same seed/k/epsilon). Record both in
  the task report.
- [ ] **Step 3: Fallback.** Force the Vulkan path (a `MIMIR_LOD_NO_CUB=1` env override in the
  selection, or an artificially tiny free-VRAM cap) and confirm a correct render + the log says
  Vulkan scatter. Add the env override as the documented escape hatch.
- [ ] **Step 4: Perf.** Record `lod ms` CUB vs Vulkan at 300M (and 1B if VRAM allows) for PT and
  raster; expect single-digit ms on CUB. Put the numbers in the task report.
- [ ] **Step 5: Docs.** Note the CUB fast path + `MIMIR_LOD_NO_CUB` in `rr-server --help`/README and
  the `pt_lod_cells`/`lod_centroid` option docs; mark the plan complete.
- [ ] **Step 6: Commit**
```bash
git add -A
git commit -m "feat(lod): raster lod timing + CUB parity/fallback/perf verification + docs"
```

---

## Self-Review

- **Spec coverage:** selection (T3), CUB pipeline (T1), interop buffers (T2), PT integration (T4),
  raster integration (T5), determinism/parity + fallback + perf + docs (T6). All spec sections map
  to a task.
- **Placeholders:** the CUB pipeline gives concrete CUB calls; integration tasks give exact
  functions/anchors and require reading the named existing patterns (interop.cpp, renderFrame) — no
  "TODO/handle later". The literal interop-export and frame-wait code is specified as "mirror the
  existing pattern at <file>" rather than invented, because it depends on internals the implementer
  must read to match exactly.
- **Type consistency:** `LodCub::reduce` signature, `reduceCuda`/`occupiedFromCuda`/`usesCub` names,
  and `reducedPositionsDevicePtr`/`occupiedDevicePtr` accessors are used consistently across T1–T5.
- **Risk ordering:** T1 (pure CUDA, unit-tested) → T2/T3 (buffers/selection, no path change) → T4
  (PT, near-swap) → T5 (raster, the risky interop sync) → T6 (verify/docs). Each task is
  independently testable and leaves the tree building with the Vulkan path intact until its path is
  switched.
