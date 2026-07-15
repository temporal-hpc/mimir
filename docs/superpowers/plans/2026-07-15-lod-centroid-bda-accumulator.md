# LOD Centroid + BDA Accumulator + VRAM-Scaled Cap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace LOD cell-center placement with mass-centroid placement (deterministic int64 fixed-point atomics), move the per-cell accumulator off the ~4 GiB descriptor cap onto a buffer-device-address pointer, and VRAM-scale the `--lod N` cap so big-memory cards run finer grids.

**Architecture:** All changes stay inside the existing LOD stage — the two Slang compute shaders, the accumulator buffers, `bindScene` sizing, and one device-feature enable. The AS build / readback / trace machinery and the `--lod 0` per-particle path are untouched. Gated on a first-task feasibility spike for int64 atomic-add through a BDA pointer, with a documented descriptor fallback.

**Tech Stack:** C++17, Vulkan 1.3 (KHR ray tracing, buffer-device-address, `shaderBufferInt64Atomics`), Slang compute shaders (runtime-compiled), CUDA interop positions.

## Global Constraints

- Centroid = `sum(positions)/count`, placed at the mass center. Determinism is mandatory: accumulate the position sum with **integer fixed-point atomics** (never float — float atomic add is non-associative → nondeterministic). Sum components are **`uint64`**; quantize each axis as `q = uint64((clamp(p,-1,1) + 1) * 0.5 * SCALE)` with `SCALE = 2^30` (a named `constexpr`, shared value between scatter and emit).
- Per-cell accumulator: `uint32 count` + `uint64 sumX,sumY,sumZ`. Count and sum buffers are reached by **BDA pointer** (push-constant `VkDeviceAddress`, `wantAddress=true`), NOT descriptors — the descriptor `maxStorageBufferRange` (~4 GiB) is exactly what we are escaping. BDA accesses use explicit 64-bit address arithmetic + cast, never pointer indexing (NVIDIA truncates `OpPtrAccessChain` past 4 GiB).
- Requires device features `shaderBufferInt64Atomics` (on `VkPhysicalDeviceVulkan12Features`) and `shaderInt64` (on `VkPhysicalDeviceFeatures`). Both must be **support-checked** before enabling (`vkCreateDevice` rejects unsupported requested features). If unsupported → LOD falls back to cell-center placement with a warning; never fail device creation.
- `--lod N` cap becomes VRAM-scaled: reject N whose accumulator (`N^3 * 32 B`) exceeds a safe fraction of device-local memory, reporting the largest feasible N. Keep a hard upper bound of 4096.
- Determinism preserved end to end: identical occupied-cell count AND centroids across repeated runs at fixed N. `--lod 0` byte-for-byte unchanged.
- Emitted occupied-cell COUNT must be identical to the shipped cell-center version at the same N (only sphere PLACEMENT changes, not which cells are occupied).
- Build: library `./mimir-build-from-change.sh`; sample relink `./samples-build-from-change.sh --sample remote-rendering`. Harmless clock-skew warnings may appear. Builds are Release (NDEBUG), so Vulkan validation layers are OFF — functional evidence is determinism + rendered image, not validation output.

### Reference run commands
Fast dev smoke (~2^20 particles, headless — server logs without a client):
```bash
samples/remote-rendering/build/rr-server 9000 1920 1080 $((2**20)) 413111 10000 \
  --pcolor 1.0,0.05,0.05 --background 0.2 --k 64 --epsilon 0.07 \
  --light-model path-tracing --spp 1 --size 0.1 --steps-per-frame 1 --fps 60 --fly --bounces 4 --lod 32
```
The shipped cell-center build emits **13689** occupied cells at `--lod 32` / 2^20; centroid must emit the SAME count (placement differs, occupancy does not). Visual/motion checks require a connected `rr-client` at 2^29 and are the user's manual follow-up.

---

## File Structure

- `lib/src/device.cpp` — support-check + enable `shaderBufferInt64Atomics`, `shaderInt64`; expose an `int64_atomics_available` capability.
- `lib/include/private/mimir/*.hpp` (device/engine/raytracing) — carry the capability flag to the RT context.
- `shaders/lod_atomic_spike.slang` — NEW (Task 1 only): minimal known-answer int64 BDA atomic test. May be deleted at the end of Task 1 or kept as a diagnostic behind an env flag.
- `shaders/pathtrace_lod_scatter.slang` — add int64 fixed-point sum accumulation.
- `shaders/pathtrace_lod_emit.slang` — place sphere at centroid; add cell-center fallback when centroid disabled.
- `lib/include/private/mimir/raytracing.hpp` — new `lod_cellsum_buffer`, `lod_centroid` flag, `LOD_FIXEDPOINT_SCALE`, push-struct fields.
- `lib/src/raytracing.cpp` — allocate the BDA sum buffer, pass BDA addresses in push constants (drop the accumulator descriptors), gate centroid vs cell-center on the capability.
- `samples/remote-rendering/rr-server.cu` — VRAM-scaled cap replacing the hardcoded 512.
- `samples/remote-rendering/README.md` — document centroid, the feature requirement, and the new cap.

---

## Task 1: Device features + int64 BDA-atomics feasibility spike (GATING)

Enable the int64-atomics device features (support-checked, with a capability flag), then prove that a 64-bit integer atomic-add through a buffer-device-address pointer compiles and produces a correct, deterministic result on this Slang/driver/GPU. This decides whether Task 2 uses the BDA accumulator (primary) or the descriptor fallback.

**Files:**
- Modify: `lib/src/device.cpp:394-435` (feature enable) and the device-suitability/query path (~205-216)
- Modify: the struct that carries device capabilities to the RT context (grep how `accel_props` / RT capability reach `RayTracingContext`; add a `bool int64_atomics` alongside)
- Create: `shaders/lod_atomic_spike.slang`
- Modify: `lib/src/raytracing.cpp` (temporary spike dispatch behind `getenv("MIMIR_LOD_ATOMIC_SPIKE")`)

**Interfaces:**
- Produces: `bool RayTracingContext::int64_atomics` (true when both features enabled); **the validated Slang snippet for int64 atomic-add through a BDA `uint64*` pointer** (documented in the Task 1 report and reused verbatim by Task 2); a definitive spike result (works / does-not-work → Task 2 branch).

- [ ] **Step 1: Support-check and enable the features**

In `lib/src/device.cpp`, where the device feature structs are populated (~line 394-430), query support first. There is already a `vkGetPhysicalDeviceFeatures2` path (~209); mirror it to read `VkPhysicalDeviceShaderAtomicInt64Features` (or read the promoted `VkPhysicalDeviceVulkan12Features::shaderBufferInt64Atomics`) and `VkPhysicalDeviceFeatures::shaderInt64`. Then, only inside the `supportsRayTracing(gpu)` block (so it rides the existing pNext chain), enable when supported:
```cpp
    // int64 buffer atomics for deterministic LOD-centroid position sums (see LOD centroid spec).
    // Support-checked: vkCreateDevice rejects unsupported requested features.
    if (int64_atomics_supported) {
        device_features.shaderInt64        = VK_TRUE;
        vk12features.shaderBufferInt64Atomics = VK_TRUE;
        spdlog::info("int64 buffer atomics enabled (LOD centroid available)");
    } else {
        spdlog::warn("int64 buffer atomics unsupported; LOD will use cell-center placement");
    }
```
Where `int64_atomics_supported` is the result of the support query. Record this capability where the device hands ray-tracing properties to the engine/RT context (grep `accel_props` assignment) so it reaches `RayTracingContext::int64_atomics`.

- [ ] **Step 2: Add the capability flag to the RT context**

In `lib/include/private/mimir/raytracing.hpp`, near `lod_cells`:
```cpp
    bool int64_atomics = false; // device supports shaderBufferInt64Atomics+shaderInt64 (LOD centroid)
```
Set it from the device capability wherever the RT context is initialized with device properties.

- [ ] **Step 3: Write the spike shader**

Create `shaders/lod_atomic_spike.slang`. The GOAL of this task is to find the Slang form that lowers to a 64-bit `OpAtomicIAdd` on a `PhysicalStorageBuffer` pointer. Try, in order, until one produces the correct sum:
(a) Slang `Atomic<uint64_t>` accessed through a pointer;
(b) `InterlockedAdd` on a dereferenced `uint64_t*`;
(c) a `spirv_asm { OpAtomicIAdd ... }` block on the pointer.
Skeleton (fill the atomic with the working form):
```slang
struct PushConstants { uint64_t* sink; uint addend; uint count; };
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64,1,1)]
void spikeMain(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= pc.count) { return; }
    // Atomically add `addend` (as u64) to *sink. Use explicit 64-bit address (single element here,
    // so offset 0). Replace the next line with the pattern that compiles + gives the exact sum.
    uint64_t* p = pc.sink;               // element 0
    /* validated int64 atomic add of uint64_t(pc.addend) into *p */
}
```

- [ ] **Step 4: Wire a temporary known-answer spike dispatch**

In `lib/src/raytracing.cpp`, behind `if (std::getenv("MIMIR_LOD_ATOMIC_SPIKE"))` at RT init (only when `int64_atomics`): create a tiny HOST_VISIBLE `uint64` BDA buffer initialized to 0, a compute pipeline over `lod_atomic_spike.slang`, dispatch `count = 100000` threads each adding `addend = 7`, `submit()` + wait, map and read. Log `spdlog::info("LOD atomic spike: got {} expected {}", got, 700000ull)`. Correct + repeatable across two runs ⇒ BDA int64 atomics work.

- [ ] **Step 5: Build and run the spike**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering`
Then: `MIMIR_LOD_ATOMIC_SPIKE=1 <fast smoke command>` (from Reference run commands).
Expected on success: `LOD atomic spike: got 700000 expected 700000`, identical on a second run.
If no candidate pattern yields 700000 (or it is nondeterministic), STOP and report the result: Task 2 will use the descriptor fallback instead.

- [ ] **Step 6: Record the outcome and commit**

Document in the Task 1 report: which atomic pattern worked (exact Slang lines) OR that none did; the observed spike values (both runs); and the resulting Task 2 branch (BDA vs descriptor). Leave the spike shader in place (harmless, env-gated) or delete it — note which.
```bash
git add lib/src/device.cpp lib/include/private/mimir/raytracing.hpp shaders/lod_atomic_spike.slang lib/src/raytracing.cpp
git commit -m "feat(lod): enable int64 buffer atomics + BDA-atomic feasibility spike"
```

---

## Task 2: Centroid accumulator + placement

Add the per-cell int64 position sum, accumulate it in scatter, and place each occupied cell's sphere at the centroid in emit — gated on `int64_atomics` (cell-center fallback otherwise). Uses the accumulator binding chosen by Task 1 (BDA primary; descriptor fallback).

**Files:**
- Modify: `shaders/pathtrace_lod_scatter.slang`, `shaders/pathtrace_lod_emit.slang`
- Modify: `lib/include/private/mimir/raytracing.hpp` (sum buffer, scale constant, push-struct fields)
- Modify: `lib/src/raytracing.cpp` (bindScene alloc + push constants + gating; drop accumulator descriptors on the BDA path)

**Interfaces:**
- Consumes: `int64_atomics` and the validated atomic pattern (Task 1); existing `lod_cellcount_buffer`, `lod_counter_buffer`, `recordLodUpdate`, `LodScatterPush`, `LodEmitPush`.
- Produces: `RtBuffer lod_cellsum_buffer` (`3 * N^3 * uint64`, DEVICE_LOCAL, BDA); `static constexpr double LOD_FIXEDPOINT_SCALE = 1073741824.0; // 2^30`; extended push structs carrying the count+sum BDA addresses and a `centroid` flag.

- [ ] **Step 1: Add the scale constant, sum buffer, and push fields (raytracing.hpp)**
```cpp
    RtBuffer lod_cellsum_buffer;   // 3 * N^3 uint64 fixed-point position sums (DEVICE_LOCAL, BDA)
    static constexpr double LOD_FIXEDPOINT_SCALE = 1073741824.0; // 2^30; maps [-1,1]->[0,2^30]
```
Extend `LodScatterPush` to carry the count and sum buffers as BDA `VkDeviceAddress`es (dropping the descriptor for them) and a `uint32 centroid` flag; extend `LodEmitPush` with the count+sum BDA addresses and the `centroid` flag. (On the descriptor fallback, keep the count/sum as descriptors and pass only the flag — see the fallback note at the end of this task.)

- [ ] **Step 2: Scatter — accumulate the int64 position sum (BDA path)**

In `shaders/pathtrace_lod_scatter.slang`, add the count + 3 sum buffers as BDA pointers in the push constants and, after computing `lin`, when `centroid != 0`:
```slang
    // Count (always).
    { uint64_t a = uint64_t(pc.cellCounts) + uint64_t(lin) * 4ull; /* atomic add 1u at *(uint*)a */ }
    if (pc.centroid != 0u) {
        float n30 = float(pc.scale);
        uint64_t qx = uint64_t((clamp(px,-1.0,1.0) + 1.0) * 0.5 * SCALE);
        uint64_t qy = uint64_t((clamp(py,-1.0,1.0) + 1.0) * 0.5 * SCALE);
        uint64_t qz = uint64_t((clamp(pz,-1.0,1.0) + 1.0) * 0.5 * SCALE);
        // sum layout: [3*lin + 0..2] uint64. Explicit 64-bit addresses; atomic-add per component
        // using the pattern validated in Task 1.
        uint64_t base = uint64_t(pc.cellSums) + uint64_t(lin) * 24ull;
        /* atomicAdd qx at base+0; qy at base+8; qz at base+16 (validated int64 BDA atomic) */
    }
```
`SCALE` is `2^30` as a shader constant matching `LOD_FIXEDPOINT_SCALE`. Keep the count atomic exactly as it is today (behavior unchanged when `centroid==0`).

- [ ] **Step 3: Emit — place at centroid (BDA path)**

In `shaders/pathtrace_lod_emit.slang`, for an occupied cell, when `centroid != 0` replace the cell-center formula with:
```slang
    uint  c  = cellCounts[lin]; // via the count BDA read
    uint64_t base = uint64_t(pc.cellSums) + uint64_t(lin) * 24ull;
    uint64_t sx = *(uint64_t*)(base + 0);
    uint64_t sy = *(uint64_t*)(base + 8);
    uint64_t sz = *(uint64_t*)(base + 16);
    // centroid in [-1,1]: -1 + 2 * (sum/count)/SCALE
    double inv = 1.0 / (double(c) * SCALE);
    float3 center = float3(-1.0) + 2.0 * float3(float(double(sx)*inv), float(double(sy)*inv), float(double(sz)*inv));
```
When `centroid == 0`, keep the existing cell-center formula unchanged.

- [ ] **Step 4: bindScene — allocate the sum buffer and switch bindings (raytracing.cpp)**

When `lod_cells > 0`: set `bool lod_centroid = int64_atomics;` (member). Allocate `lod_cellsum_buffer` sized `3 * cells * sizeof(uint64_t)` with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT`, `DEVICE_LOCAL`, `wantAddress=true` — only when `lod_centroid`. Allocate `lod_cellcount_buffer` with `wantAddress=true` now (BDA). Remove the scatter/emit accumulator DESCRIPTOR writes for count/sum (the counter stays a descriptor in emit, or also move to BDA — keep the counter as-is to minimize change). Pass the count+sum BDA addresses and `centroid` flag in the push constants in `recordLodUpdate` (scatter and emit), and clear `lod_cellsum_buffer` with `vkCmdFillBuffer(...,0)` alongside the count buffer. Log `LOD placement: centroid` or `cell-center (int64 atomics unavailable)`.

- [ ] **Step 5: Build, verify determinism + equal count**

Build both. Run the fast smoke `--lod 32`:
- Expect the emitted count **13689** (UNCHANGED from cell-center — occupancy is identical), logged `LOD placement: centroid`.
- Run twice; the count must be identical (determinism).
- Run `--lod 0`; unchanged per-particle path.
If the count differs from 13689, the aggregation topology changed (a bug) — stop and diagnose.

- [ ] **Step 6: Commit**
```bash
git add shaders/pathtrace_lod_scatter.slang shaders/pathtrace_lod_emit.slang \
        lib/include/private/mimir/raytracing.hpp lib/src/raytracing.cpp
git commit -m "feat(lod): centroid placement via int64 fixed-point BDA accumulator"
```

**Descriptor fallback (only if Task 1's spike failed):** keep `lod_cellcount_buffer` and a new `lod_cellsum_buffer` as `RWStructuredBuffer` DESCRIPTORS (`uint` and `uint64`), using `InterlockedAdd` on structured-buffer elements (no BDA). This caps centroid at `N ≤ 511` (32 B/cell under `maxStorageBufferRange`); Task 3 must then keep 511 as the hard cap when `lod_centroid` is on. Everything else in this task is identical. Document in the report which path was taken.

---

## Task 3: VRAM-scaled `--lod` cap

Replace the hardcoded `--lod N > 512` rejection with a limit derived from device-local memory, so big-VRAM cards run finer grids and over-budget N is rejected with the largest feasible N.

**Files:**
- Modify: `samples/remote-rendering/rr-server.cu:343-345` (the current cap) and near the VRAM query it already does
- Optionally: `lib/src/raytracing.cpp` bindScene (a defensive library-side check)

**Interfaces:**
- Consumes: the accumulator size rule (`bytes = N^3 * (lod_centroid ? 32 : 4)`); device-local VRAM (the server already gathers `cudaMemGetInfo` free/total, e.g. `vram_free0`).

- [ ] **Step 1: Compute the cap from available VRAM**

In `rr-server.cu`, replace the `if (lod_cells > 512)` block with a limit computed from free device-local memory gathered at setup. Reserve headroom for the particle + AABB buffers; use a safe fraction (e.g. accumulator ≤ 50% of free VRAM). Per-cell bytes = 32 when centroid is available else 4 (query the instance/option for whether int64 atomics are on, or conservatively assume 32). Reject with the largest feasible N:
```cpp
    // Accumulator is N^3 * bytes_per_cell (32 for centroid, 4 for cell-center). Bound N so it fits a
    // safe fraction of device-local VRAM; keep a hard sanity ceiling.
    const unsigned long long bytes_per_cell = 32ull; // conservative (centroid)
    const unsigned long long budget = vram_free0 / 2ull;
    unsigned long long max_cells = budget / bytes_per_cell;
    unsigned int max_n = 4096;
    while ((unsigned long long)max_n*max_n*max_n > max_cells) { --max_n; }
    if (lod_cells > max_n) {
        fprintf(stderr, "rr-server: --lod %u exceeds VRAM budget; max feasible N is %u "
                        "(accumulator %.1f GB)\n", lod_cells, max_n,
                        (double)((unsigned long long)lod_cells*lod_cells*lod_cells*bytes_per_cell)/1e9);
        return EXIT_FAILURE;
    }
```
(If Task 2 took the descriptor fallback, additionally clamp `max_n` to 511.)

- [ ] **Step 2: Diminishing-returns warning**

After acceptance, if `(unsigned long long)lod_cells^3 > point_count/8` (occupied approaching P), print an info line that LOD gives little benefit in this zone.

- [ ] **Step 3: Update the usage text**

Change the `--lod` usage line from "0..512" to describe the VRAM-scaled cap (e.g. "0..VRAM-limited; larger N needs more memory").

- [ ] **Step 4: Build and verify the cap**

Build the sample. On the 95 GB card: confirm an N that fits above 512 (e.g. `--lod 640`) is accepted and runs, and an over-budget N (e.g. `--lod 4096`) is rejected with the "max feasible N is ..." message. Confirm `--lod 512` still works.

- [ ] **Step 5: Commit**
```bash
git add samples/remote-rendering/rr-server.cu
git commit -m "feat(lod): VRAM-scaled --lod cap replacing the fixed 512"
```

---

## Task 4: Documentation

Update the README for centroid placement, the int64-atomics requirement + cell-center fallback, and the VRAM-scaled cap.

**Files:**
- Modify: `samples/remote-rendering/README.md` (the `--lod` section from the prior feature)

- [ ] **Step 1: Update the `--lod` docs**

Amend the existing `--lod` section: spheres are placed at the per-cell **mass centroid** (follows the cloud, smooth motion) when the GPU supports `shaderBufferInt64Atomics`, else at the cell center (a warning is logged); the cap is now **VRAM-scaled** (N bounded by device memory, not a fixed 512); note the accumulator cost (`N^3 * 32 bytes` for centroid) and that determinism is preserved. Keep the memory table but update it for 32 B/cell.

- [ ] **Step 2: Commit**
```bash
git add samples/remote-rendering/README.md
git commit -m "docs(lod): centroid placement, int64 requirement, VRAM-scaled cap"
```

---

## Self-Review Notes

- **Spec coverage:** centroid via int64 fixed-point (Task 2 shaders + constant), BDA accumulator escaping the descriptor cap (Task 1 spike → Task 2 BDA buffers), device-feature enable with support-check + cell-center fallback (Task 1 Step 1, Task 2 gating), VRAM-scaled cap + diminishing-returns warning (Task 3), determinism (Task 2 Step 5 equal-count/repeat check), `--lod 0` unchanged (Task 2 Step 5), docs (Task 4). Fallback path documented (Task 2 note + Task 3 clamp).
- **Placeholder scan:** the only deferred specifics are the exact Slang int64-BDA-atomic call (Task 1's deliverable, consumed by Task 2 — a genuine cross-task interface, not a placeholder) and the descriptor-fallback shader form (documented, used only on spike failure). All host-side steps carry concrete code.
- **Type consistency:** `int64_atomics`, `lod_centroid`, `lod_cellsum_buffer`, `LOD_FIXEDPOINT_SCALE` / shader `SCALE = 2^30`, and the extended `LodScatterPush`/`LodEmitPush` fields are named identically across Tasks 1-3.
- **Risks to watch during execution:** (a) Task 1 is genuinely exploratory — if no int64-BDA-atomic pattern works, execution branches to the descriptor fallback (N ≤ 511) and Task 3 clamps accordingly; (b) confirm the emitted occupied-cell count stays 13689 at `--lod 32` (a change means a topology regression, not a placement change); (c) the `double`-precision centroid divide in emit needs `shaderFloat64` (already enabled at device.cpp:399) — verify it survives Slang compilation; (d) clearing the larger sum buffer each frame adds fill bandwidth — negligible vs the build, but confirm no barrier gap.
