# 64-bit Particle Counts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let mimir render more than 2^32 particles (bounded only by GPU memory) across all four light models, with or without LOD.

**Architecture:** Approach A — the *total* element count becomes 64-bit and flows through the public API and engine; every per-operation count that Vulkan caps at `uint32` (raster draw, BLAS geometry, compute invocation id) stays `uint32` and gets a loop driven by the 64-bit total. Buffers are already `VkDeviceSize`/`size_t`, so only counts and three chunk loops change.

**Tech Stack:** C++/Vulkan (mimir library), Slang compute shaders (runtime-compiled), CUDA/Vulkan interop, the remote-rendering + particles-kmodal-3d samples. No GPU unit-test harness — verification is build + headless/`drive-the-app` runs with expected log lines.

**Spec:** `docs/superpowers/specs/2026-07-16-64bit-particle-counts-design.md`

## Global Constraints

- **Approach A:** one 64-bit total (`element_count`); per-operation counts stay `uint32`; chunk at the three Vulkan boundaries. Never widen `draw_count`/`instance_count` semantics beyond per-chunk/bounded values.
- **Raster chunk `CAP = UINT32_MAX` (2^32-1).** Chunking is a pure extension: counts ≤ 2^32-1 (including the 2^31–2^32 band) must render as a **single draw, byte-for-byte unchanged**; only counts > 2^32-1 chunk. The chunked draw must be **seamless** (no gap at chunk boundaries).
- **No user-facing count cap.** Remove the interim `>= 2^32` guard; keep the `count == 0` reject; replace the cap with a **memory pre-flight** (query live free VRAM via `cudaMemGetInfo`, reject over-budget with "needs X GB, only Y free").
- **Determinism preserved:** the compute grid-stride visits every particle exactly once, so LOD occupied counts/centroids are identical to today; `--lod 0` and all light models are behavior-preserving at counts ≤ 2^32-1.
- **Unchanged limits:** the LOD grid cap `N ≤ 1625` (uint32 cell index) and the emit/finalize passes (1-D, over N^3 < 2^32) do not change.
- **`shaderInt64`** is required for the PT/LOD compute paths (already enabled inside the `supportsRayTracing` block in `device.cpp`); do not add a new device requirement for the pure-raster path.
- **BDA access rule (existing):** in shaders, compute the byte address in explicit 64-bit integer arithmetic and cast — NEVER index a BDA pointer (`p[i]`), which the NVIDIA compiler truncates to 32-bit past 4 GiB. All new shader indexing must follow this.

---

## File Structure

- `lib/include/public/mimir/view.hpp` — `Layout` (x/y/z, `getTotalCount`, `make`) and `AttributeDescription::size` widen to 64-bit (public API break).
- `lib/include/private/mimir/api.hpp` — `View` gains `uint64_t element_count` and per-binding `vbo_stride[]`/`vbo_rate[]` (for chunked rebinding).
- `lib/src/engine.cpp` — `getDrawCount` returns 64-bit; `createView` sets `element_count` + strides/rates; the mesh setup and PT `bindScene` calls use `element_count`; `drawElements` gets the raster chunk loop.
- `lib/include/private/mimir/raytracing.hpp` + `lib/src/raytracing.cpp` — `bindScene`/`particle_count`/BLAS-chunk index math widen to 64-bit; the AABB-writer dispatch becomes grid-stride.
- `shaders/pathtrace_aabbs.slang`, `shaders/pathtrace_lod_scatter.slang` — 64-bit count + grid-stride loop.
- `lib/src/lod.cpp` — `LodScatterPush` 64-bit count + stride; grid-stride dispatch.
- `samples/remote-rendering/rr-server.cu`, `samples/particles-kmodal-3d/benchmark_mimir.cpp`, `samples/particles-kmodal-3d/kmodal_sim.cuh` — 64-bit `point_count`/`PointsParams::count`, guard removal, memory pre-flight.
- `samples/*/README.md` + rr-server `--help` — docs.

---

## Task 1: Widen the public API + view count types to 64-bit (no chunking)

**Deliverable:** `Layout`/`AttributeDescription::size` are 64-bit, the `View` carries a 64-bit `element_count`, and everything compiles and renders identically for counts ≤ 2^32-1. No chunking yet (a count > 2^32-1 would still truncate at the draw — Task 3 fixes that — but Task 1 introduces no wrap in the *plumbing*).

**Files:**
- Modify: `lib/include/public/mimir/view.hpp` (Layout, AttributeDescription)
- Modify: `lib/include/private/mimir/api.hpp` (View)
- Modify: `lib/src/engine.cpp` (`getDrawCount`, `createView`, mesh setup ~1316-1326, PT `bindScene` calls ~222-271)

**Interfaces:**
- Produces: `Layout{size_t x,y,z}`, `Layout::make(size_t,size_t,size_t)`, `Layout::getTotalCount()->size_t`, `AttributeDescription::size` is `size_t`, `View::element_count` is `uint64_t`, `getDrawCount(ViewDescription*)->uint64_t`.

- [ ] **Step 1: Widen `Layout` and `AttributeDescription::size` in `view.hpp`**

In `lib/include/public/mimir/view.hpp`, change `Layout`:
```cpp
struct Layout
{
    size_t x = 0, y = 1, z = 1;
    size_t getTotalCount() { return x * y * z; };
    static Layout make(size_t x, size_t y = 1, size_t z = 1)
    {
        return Layout{ .x = x, .y = y, .z = z };
    }
};
```
And change `AttributeDescription::size` from `unsigned int size = 0;` to `size_t size = 0;`. Leave `IndexDescription::size` and `index_size` as `unsigned int` (index buffers are the small icosphere; not the >2^32 path).

- [ ] **Step 2: Add `element_count` to the `View` struct**

In `lib/include/private/mimir/api.hpp`, inside `struct View`, add after `instance_count`:
```cpp
    // True element (particle/vertex) total — 64-bit source of truth for the chunk loops.
    // draw_count/instance_count stay uint32 (per-chunk or the mesh index count).
    uint64_t element_count = 0;
```

- [ ] **Step 3: Make `getDrawCount` return 64-bit and set `element_count` at view creation**

In `lib/src/engine.cpp`, change `getDrawCount`:
```cpp
uint64_t getDrawCount(ViewDescription *desc)
{
    auto& pos_attr = desc->attributes[AttributeType::Position];
    return hasIndexing(pos_attr)? pos_attr.indexing.size : pos_attr.size;
}
```
In `createView`, set both fields in the `View{...}` initializer — `element_count` holds the total; `draw_count` holds a bounded uint32 (only meaningful for the mesh index count, overwritten in Step 4; for point markers it is unused once Task 3 lands, but keep it valid):
```cpp
    View view{
        .pipeline    = VK_NULL_HANDLE,
        .draw_count  = static_cast<uint32_t>(std::min<uint64_t>(getDrawCount(desc), UINT32_MAX)),
        // ... existing fields ...
    };
    view.element_count = getDrawCount(desc);
```
(Place `view.element_count = getDrawCount(desc);` right after the `View view{...}` initializer, before the body uses it. Ensure `<algorithm>` and `<cstdint>` are included in engine.cpp — add if missing.)

- [ ] **Step 4: Use `element_count` in the mesh setup and PT bindScene calls**

In `lib/src/engine.cpp` mesh setup (~1316-1326), the particle total must come from `element_count`, not `draw_count`:
```cpp
        uint64_t particle_count = view.element_count;   // was view.draw_count
        // ... ensureSphereMesh() etc ...
        view.draw_count     = sphere_index_count;       // icosphere index count (uint32, small)
        view.instance_count = static_cast<uint32_t>(std::min<uint64_t>(particle_count, UINT32_MAX));
        // element_count already holds the true particle total; the Task 3 chunk loop reads it.
```
In the PT/raster LOD init blocks that read `view->draw_count` as the particle count (engine.cpp ~222, ~226, ~271), replace with `view->element_count`:
- `bindScene(pos_addr, view->element_count, view->desc.default_size, ...)` (the `bindScene` signature widens in Task 4; for now it still takes `uint32_t`, so pass `static_cast<uint32_t>(std::min<uint64_t>(view->element_count, UINT32_MAX))` here and note it — Task 4 removes the cast).
- LOD raster init (~271): `uint64_t particle_count = view->element_count;` (drop the `is_mesh ? instance_count : draw_count` expression — `element_count` is the total in both cases). `LodContext::init`/`recordReduction` still take `uint32_t` here; Task 5 widens them, so for now cast: `static_cast<uint32_t>(std::min<uint64_t>(view->element_count, UINT32_MAX))`.

- [ ] **Step 5: Build the library and both samples**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering && cmake --build samples/particles-kmodal-3d/build --target benchmark_mimir -j"$(nproc)"`
Expected: all compile (samples pass `unsigned` literals to `Layout::make`/`size` — they promote to `size_t` cleanly). Fix any narrowing-conversion warnings the widened types surface in the samples' `createView` calls.

- [ ] **Step 6: Verify ≤2^32 behavior is unchanged (headless)**

Run: `samples/remote-rendering/build/rr-server 9100 512 512 $((2**20)) 0 --light-model path-tracing --lod 32 2>&1 | grep -iE "LOD emitted|particles|occupied"`
Expected: `LOD emitted 1472 occupied cells (reduction 712:1 vs 1048576 particles)` — the count is logged correctly and the LOD occupied count is unchanged from before this task.
Run: `samples/remote-rendering/build/rr-server 9101 512 512 $((2**31)) 0 2>&1 | grep -iE "particles|VRAM"`
Expected: `... 2147483648 points ...` and a successful VRAM setup line (no wrap, no crash) — same as the pre-change build.

- [ ] **Step 7: Commit**

```bash
git add lib/include/public/mimir/view.hpp lib/include/private/mimir/api.hpp lib/src/engine.cpp
git commit -m "feat(64bit): widen Layout/AttributeDescription counts + add View::element_count"
```

---

## Task 2: Samples + sim to 64-bit, remove the 2^32 guard, add the memory pre-flight

**Deliverable:** a count > 2^32-1 is accepted and logged with the correct value (no wrap), and an over-memory count is rejected up front with a clear message instead of an `IOT`/OOM abort. (Rendering of >2^32 is still wrong until Task 3 — this task only proves the *count plumbing + pre-flight*.)

**Files:**
- Modify: `samples/particles-kmodal-3d/kmodal_sim.cuh` (`PointsParams::count`)
- Modify: `samples/remote-rendering/rr-server.cu` (point_count type, guard→pre-flight)
- Modify: `samples/particles-kmodal-3d/benchmark_mimir.cpp` (pts.count type, guard→pre-flight)

**Interfaces:**
- Consumes: `Layout`/`AttributeDescription::size` are `size_t` (Task 1).
- Produces: `PointsParams::count` is `uint64_t`; both samples parse a 64-bit count and run a `preflightVram(count, has_lod, is_pt)` check.

- [ ] **Step 1: Widen `PointsParams::count` in the shared sim header**

In `samples/particles-kmodal-3d/kmodal_sim.cuh`, change `unsigned int count = 1'000'000;` to `uint64_t count = 1'000'000;` (add `#include <cstdint>` if absent). The kernels already take `size_t point_count` and grid-stride, so no kernel change is needed. In `kmodal_sim.cu`, the `d_ids` allocation `cudaMalloc(&d_ids, sizeof(unsigned int) * params.count)` now multiplies by a 64-bit count — verify it is `sizeof(unsigned int) * (size_t)params.count` (add the cast if the multiply would be 32-bit).

- [ ] **Step 2: rr-server — 64-bit point_count + remove the 2^32 guard**

In `samples/remote-rendering/rr-server.cu`, change `unsigned int point_count = 100000;` to `uint64_t point_count = 100000;`. Replace the positional-parse guard block (the `if (posv.size() >= 4) { ... point_count must be in [1, 4294967295] ... }` added earlier) with a 64-bit parse that keeps ONLY the `== 0` reject:
```cpp
    if (posv.size() >= 4) {
        unsigned long long pc = std::stoull(posv[3]);
        if (pc == 0ull) {
            fprintf(stderr, "rr-server: point_count must be >= 1\n");
            return EXIT_FAILURE;
        }
        point_count = (uint64_t)pc;   // no upper cap: the memory pre-flight below bounds it
    }
```
`pts.count = point_count;` stays (both are 64-bit now).

- [ ] **Step 3: rr-server — memory pre-flight replacing the LOD-only VRAM check**

In `samples/remote-rendering/rr-server.cu`, after `cudaMemGetInfo(&vram_free0, &vram_total)` and after `lod_cells`/`light_model` are known, ADD a per-particle memory pre-flight. This is **additive** — the existing LOD-accumulator check (N^3 × 32 B) and its `N <= 1625` clamp size the grid and stay exactly as-is; the pre-flight sizes the *per-particle* buffers (positions, and AABBs under PT-no-LOD), a different allocation. Add:
```cpp
    // Memory pre-flight: reject a count that will not fit the GPU memory free right now, BEFORE Vulkan
    // OOMs. Dominant device allocations: positions (12 B/particle, always) + per-particle AABBs
    // (24 B/particle) only under path-tracing WITHOUT LOD (LOD builds the BVH over occupied cells).
    {
        const bool pt_no_lod = (light_model == LightModel::PathTracing) && (lod_cells == 0);
        const unsigned long long bytes_per_particle = 12ull + (pt_no_lod ? 24ull : 0ull);
        const unsigned long long need = (unsigned long long)point_count * bytes_per_particle;
        if (need > (unsigned long long)vram_free0) {
            fprintf(stderr, "rr-server: %llu particles need %.1f GB (%s) but only %.1f GB is free on "
                    "the GPU right now\n", (unsigned long long)point_count, (double)need/1e9,
                    pt_no_lod ? "positions+AABBs" : "positions", (double)vram_free0/1e9);
            return EXIT_FAILURE;
        }
    }
```
(The LOD-accumulator VRAM/1625 check for `lod_cells > 0` stays as-is; this pre-flight is additive and covers the per-particle buffers.)

- [ ] **Step 4: benchmark_mimir — same 64-bit count, guard removal, pre-flight**

In `samples/particles-kmodal-3d/benchmark_mimir.cpp`: the `PointsInput` uses `input.pts.count` which is now `uint64_t` (Task 2 Step 1). Replace the positional-parse guard (`if (pos.size() >= 3) { ... point count must be in [1, 4294967295] ... }`) with:
```cpp
    if (pos.size() >= 3) {
        unsigned long long pc = std::stoull(pos[2]);
        if (pc == 0ull) { fprintf(stderr, "benchmark_mimir: point count must be >= 1\n"); exit(EXIT_FAILURE); }
        input.pts.count = (uint64_t)pc;   // no upper cap; memory pre-flight bounds it
    }
```
In `runExperiment`, after the CUDA context is up (`cudaSetDevice`/`cudaFree` ~258-261) and before `createInstance`, add the same pre-flight (benchmark_mimir has no LOD flag yet unless it was wired — if `opts.pt_lod_cells`/`input.pt_lod` exists, use it; else treat `has_lod = (input.pt_lod > 0)`):
```cpp
    {
        size_t vram_free = 0, vram_total = 0;
        checkCuda(cudaMemGetInfo(&vram_free, &vram_total));
        const bool pt_no_lod = (input.light_model == LightModel::PathTracing) && (input.pt_lod == 0);
        const unsigned long long bytes_per_particle = 12ull + (pt_no_lod ? 24ull : 0ull);
        const unsigned long long need = (unsigned long long)n * bytes_per_particle;
        if (need > (unsigned long long)vram_free) {
            fprintf(stderr, "benchmark_mimir: %zu particles need %.1f GB but only %.1f GB free\n",
                    (size_t)n, (double)need/1e9, (double)vram_free/1e9);
            exit(EXIT_FAILURE);
        }
    }
```
(`n` is `input.pts.count`; it is currently `const size_t n = input.pts.count;` — keep it, now sourced from a 64-bit field. This pre-flight is **additive**: the earlier LOD-accumulator VRAM cap (N^3 × 32 B) and its `N<=1625` clamp size the grid and stay as-is.)

- [ ] **Step 5: Build both samples**

Run: `./samples-build-from-change.sh --sample remote-rendering && cmake --build samples/particles-kmodal-3d/build --target benchmark_mimir -j"$(nproc)"`
Expected: both compile.

- [ ] **Step 6: Verify count plumbing + pre-flight (headless)**

Run: `samples/remote-rendering/build/rr-server 9110 320 240 $((2**32 + 5)) 0 --light-model phong 2>&1 | grep -iE "points|particles|free|need"`
Expected: the log shows `4294967301 points` (NOT 5 — no wrap). It either sets up VRAM (if ~52 GB fits — at 12 B/particle 2^32+5 ≈ 51.5 GB on a 96 GB card it fits) or, if over budget, prints the clean "needs X GB, only Y free" message. **No `IOT`/abort.**
Run: `samples/remote-rendering/build/rr-server 9111 320 240 $((2**34)) 0 --light-model phong 2>&1 | grep -iE "need|free|IOT"`
Expected: 2^34 ≈ 17 B particles × 12 B = 206 GB > 96 GB → clean `needs 206.2 GB ... only ~95 GB free` rejection, no crash.
Run: `samples/remote-rendering/build/rr-server 9112 320 240 0 0 2>&1 | grep -iE "must be"`
Expected: `point_count must be >= 1`.

- [ ] **Step 7: Commit**

```bash
git add samples/particles-kmodal-3d/kmodal_sim.cuh samples/remote-rendering/rr-server.cu samples/particles-kmodal-3d/benchmark_mimir.cpp
git commit -m "feat(64bit): 64-bit sample point counts + memory pre-flight; drop the 2^32 guard"
```

---

## Task 3: Raster draw-chunking (none / phong / phong-mesh)

**Deliverable:** the non-LOD raster draws render > 2^32-1 vertices/instances by looping `vkCmdDraw`/`vkCmdDrawIndexed` over chunks of ≤ `UINT32_MAX`, rebinding each advancing vertex binding at a 64-bit byte offset. Counts ≤ 2^32-1 run a single chunk (unchanged). LOD indirect draws are untouched.

**Files:**
- Modify: `lib/include/private/mimir/api.hpp` (`View`: add `vbo_stride[]`, `vbo_rate[]`)
- Modify: `lib/src/engine.cpp` (`createView` populate strides/rates; `drawElements` chunk loop ~2440-2446)

**Interfaces:**
- Consumes: `View::element_count` (Task 1).
- Produces: the raster draw path reads `element_count` and chunks; per-binding `vbo_stride`/`vbo_rate` describe how each binding advances.

- [ ] **Step 1: Add per-binding stride + input rate to the `View` struct**

In `lib/include/private/mimir/api.hpp`, inside `struct View`, after `VkDeviceSize offsets[max_attr_count];` add:
```cpp
    // Per-binding element stride (bytes) and input rate, so the chunked draw can advance each binding
    // by chunk_start * stride for the bindings whose rate matches the chunked dimension.
    VkDeviceSize      vbo_stride[max_attr_count] = {0};
    VkVertexInputRate vbo_rate[max_attr_count]   = {VK_VERTEX_INPUT_RATE_VERTEX};
```

- [ ] **Step 2: Populate `vbo_stride`/`vbo_rate` in `createView`**

In `lib/src/engine.cpp` `createView`, wherever a vertex buffer is added (`view.vbo[view.vb_count] = ...; view.vb_count++;` at ~1239, ~1266, ~1299), set the matching stride and rate for that binding index. For the per-vertex attribute buffers the stride is the element size and the rate is `VK_VERTEX_INPUT_RATE_VERTEX`:
```cpp
    view.vbo_stride[view.vb_count] = static_cast<VkDeviceSize>(attr.format.getSize());
    view.vbo_rate[view.vb_count]   = VK_VERTEX_INPUT_RATE_VERTEX;
    view.vbo[view.vb_count] = createAttributeBuffer(vb_size, vb_usage, vb_mem);
    view.vb_count++;
```
For the SphereMesh instanced path (mesh setup ~1316-1326, where binding 0 = icosphere template per-vertex and binding 1 = per-instance centers), set binding 1's rate to instance and stride to `sizeof(glm::vec3)`:
```cpp
    // binding 0 = template (per-vertex, vec3); binding 1 = per-instance centers (vec3)
    view.vbo_stride[0] = sizeof(glm::vec3); view.vbo_rate[0] = VK_VERTEX_INPUT_RATE_VERTEX;
    view.vbo_stride[1] = sizeof(glm::vec3); view.vbo_rate[1] = VK_VERTEX_INPUT_RATE_INSTANCE;
```
(Match the exact bindings the mesh pipeline uses — see `pipeline.cpp:396-399`. If the mesh path builds its bindings elsewhere, set the two strides/rates there.)

- [ ] **Step 3: Replace the two non-LOD draws with a chunk loop**

In `lib/src/engine.cpp` `drawElements`, the non-LOD branches currently are:
```cpp
        if (view->use_ibo) { ... vkCmdDrawIndexed(cmd, view->draw_count, view->instance_count, 0, 0, 0); }
        else if (lod_point_draw) { ... }
        else { vkCmdDraw(cmd, view->draw_count, view->instance_count, first_vertex, 0); }
```
Introduce a `CAP` and chunk loops. Add near the top of the file (file scope):
```cpp
static constexpr uint64_t kDrawChunkCap = UINT32_MAX; // Vulkan vertexCount/instanceCount hard max
```
Add a helper (file scope in engine.cpp) that rebinds all bindings at a chunk offset and draws:
```cpp
// Rebind every vertex binding at chunk_start (advancing only bindings whose rate == chunk_rate) and
// issue one draw of `n` elements. chunk_rate = VK_VERTEX_INPUT_RATE_VERTEX for point clouds (chunk
// vertices) or VK_VERTEX_INPUT_RATE_INSTANCE for meshes (chunk instances).
static void drawChunk(VkCommandBuffer cmd, const View* view, uint64_t chunk_start, uint32_t n,
                      VkVertexInputRate chunk_rate, bool indexed, uint32_t index_count)
{
    VkBuffer     vbos[max_attr_count];
    VkDeviceSize offs[max_attr_count];
    for (uint32_t b = 0; b < view->vb_count; ++b) {
        vbos[b] = view->vbo[b];
        offs[b] = view->offsets[b] +
            (view->vbo_rate[b] == chunk_rate ? chunk_start * view->vbo_stride[b] : (VkDeviceSize)0);
    }
    vkCmdBindVertexBuffers(cmd, 0, view->vb_count, vbos, offs);
    if (indexed) vkCmdDrawIndexed(cmd, index_count, n, 0, 0, 0);
    else         vkCmdDraw(cmd, n, 1u, 0, 0);
}
```
Replace the two non-LOD draws so each runs the loop (the LOD `lod_point_draw`/`lod_mesh_draw` indirect branches stay exactly as they are, above these):
```cpp
        if (view->use_ibo && !lod_mesh_draw) // mesh, no LOD: chunk the INSTANCE dimension
        {
            vkCmdBindIndexBuffer(cmd, view->ibo, 0, view->index_type);
            for (uint64_t start = 0; start < view->element_count; start += kDrawChunkCap) {
                uint32_t n = (uint32_t)std::min<uint64_t>(kDrawChunkCap, view->element_count - start);
                drawChunk(cmd, view, start, n, VK_VERTEX_INPUT_RATE_INSTANCE, true, view->draw_count);
            }
        }
        else if (!view->use_ibo && !lod_point_draw) // point cloud, no LOD: chunk the VERTEX dimension
        {
            for (uint64_t start = 0; start < view->element_count; start += kDrawChunkCap) {
                uint32_t n = (uint32_t)std::min<uint64_t>(kDrawChunkCap, view->element_count - start);
                drawChunk(cmd, view, start, n, VK_VERTEX_INPUT_RATE_VERTEX, false, 0);
            }
        }
```
Keep the LOD indirect branches (`lod_point_draw` → `vkCmdDrawIndirect`, `lod_mesh_draw` → `vkCmdDrawIndexedIndirect`) unchanged and reached first. Ensure the vertex-buffer bind for the LOD branches still happens as today (those branches bind before drawing). The plain `else { vkCmdBindVertexBuffers(...) }` that used to bind for the non-LOD path is now folded into `drawChunk` — remove the now-dead standalone `vkCmdBindVertexBuffers(cmd, 0, view->vb_count, view->vbo, view->offsets)` for the non-LOD case so binding happens once per chunk.

- [ ] **Step 4: Build**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering`
Expected: compiles.

- [ ] **Step 5: Verify ≤2^32 unchanged + >2^32 renders (client-driven)**

Regression (single chunk): render a small `phong` scene headless with a client and confirm a non-blank frame:
Run: start `samples/remote-rendering/build/rr-server 9120 640 480 1048576 0 --light-model phong` and connect the headless `rr-client` (as prior tasks did), confirm a non-blank PPM. The image must look exactly as before (single draw).
>2^32 (two chunks) on the 96 GB card: `samples/remote-rendering/build/rr-server 9121 640 480 5000000000 0 --light-model none` (5 B × 12 B = 60 GB, fits). Connect a client; confirm the server logs `5000000000 points`, renders **non-blank** frames with the cloud filling the domain (no missing half — proves the second chunk drew), and disconnects cleanly. Repeat with `--light-model phong` and `--light-model phong-mesh`.

- [ ] **Step 6: Commit**

```bash
git add lib/include/private/mimir/api.hpp lib/src/engine.cpp
git commit -m "feat(64bit): raster draw-chunking over UINT32_MAX for none/phong/phong-mesh"
```

---

## Task 4: PT BLAS chunk-math → 64-bit

**Deliverable:** `bindScene`, `particle_count`, and the BLAS-chunk index math are 64-bit, so path-tracing (no LOD) builds a correct chunked BVH past 2^32. Counts ≤ 2^32-1 are unchanged.

**Files:**
- Modify: `lib/include/private/mimir/raytracing.hpp` (`particle_count`, `bindScene` decl)
- Modify: `lib/src/raytracing.cpp` (`chunkPrimCount`, `createDynamicBlasChunks`, `bindScene` body, `particle_count` uses)
- Modify: `lib/src/engine.cpp` (drop the `bindScene` cast added in Task 1 Step 4)

**Interfaces:**
- Consumes: `View::element_count` (Task 1).
- Produces: `void bindScene(VkDeviceAddress positions, uint64_t particle_count, float radius, glm::vec4 color)`; `RayTracingContext::particle_count` is `uint64_t`.

- [ ] **Step 1: Widen `particle_count` and `bindScene` in the header**

In `lib/include/private/mimir/raytracing.hpp`: change `uint32_t particle_count = 0;` to `uint64_t particle_count = 0;`. Change the `bindScene` declaration to `void bindScene(VkDeviceAddress positions, uint64_t particle_count, float radius, glm::vec4 color);`. Leave `blas_chunk_prims` (`uint32_t`) and `lod_max_cells` (`uint32_t`) as-is (a chunk and the LOD cell count are each < 2^32).

- [ ] **Step 2: Widen the chunk index math**

In `lib/src/raytracing.cpp`:
- `chunkPrimCount` (~155): compute the base in 64-bit; the returned count is still ≤ `blas_chunk_prims` (< 2^32):
```cpp
uint32_t chunkPrimCount(const RayTracingContext& ctx, uint32_t c)
{
    uint64_t base = (uint64_t)c * ctx.blas_chunk_prims;
    uint64_t rem  = ctx.particle_count - base;
    return rem < ctx.blas_chunk_prims ? (uint32_t)rem : ctx.blas_chunk_prims;
}
```
- `createDynamicBlasChunks` (~165): signature `(RayTracingContext& ctx, VkDeviceAddress aabb_addr, uint64_t count)`; `num_chunks` computed in 64-bit then stored in `uint32_t` (at 2^37 particles / 2^29 chunk ≈ 256 chunks, fits):
```cpp
void createDynamicBlasChunks(RayTracingContext& ctx, VkDeviceAddress aabb_addr, uint64_t count)
{
    uint32_t num_chunks = (uint32_t)((count + ctx.blas_chunk_prims - 1) / ctx.blas_chunk_prims);
    // ... unchanged body ...
}
```
- `chunkAabbAddr` (~149): already casts `c` to `VkDeviceSize` — no change. Its callers are unchanged.

- [ ] **Step 3: Widen `bindScene` and its `particle_count` uses**

In `lib/src/raytracing.cpp` `bindScene` (~1198): change the parameter to `uint64_t particle_count`; the body's `particle_count = count;` (~1201) and `if (particle_count == 0) return;` (~1345) are fine with 64-bit. Where `particle_count` sizes the AABB buffer / feeds `createDynamicBlasChunks`, ensure the multiply is 64-bit (`(VkDeviceSize)particle_count * sizeof(VkAabbPositionsKHR)`). The AABB-writer push `.count = particle_count` is handled in Task 5 (the shader/struct widen there); for now keep it compiling by casting `.count = (uint32_t)std::min<uint64_t>(particle_count, UINT32_MAX)` and leave a `// Task 5: 64-bit count + grid-stride` comment. Now that `RayTracingContext::particle_count` is `uint64_t`, the `recordLodUpdate` call `lod->recordReduction(c, position_address, particle_count, 0)` passes a 64-bit value to a still-`uint32_t` parameter (widened in Task 5) — add a temporary `(uint32_t)std::min<uint64_t>(particle_count, UINT32_MAX)` cast there to silence the narrowing, removed in Task 5.

- [ ] **Step 4: Drop the Task 1 cast at the engine bindScene call**

In `lib/src/engine.cpp` (~226), now that `bindScene` takes `uint64_t`, pass `view->element_count` directly (remove the `static_cast<uint32_t>(std::min...)` added in Task 1 Step 4).

- [ ] **Step 5: Build + verify ≤2^32 PT unchanged**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering`
Run: `samples/remote-rendering/build/rr-server 9130 512 512 $((2**28)) 0 --light-model path-tracing 2>&1 | grep -iE "BLAS|chunk|particles"`
Expected: 2^28 = 268435456 particles → the PT setup logs the same BLAS-chunk line as before this task (e.g. `... 1 BLAS chunk(s) ...` or the multi-chunk count for 2^28 given `blas_chunk_prims`), no crash. This exercises the 64-bit chunk math with no behavior change.
Note: PT **without LOD past 2^32** needs the B300 (154 GB of AABBs at 4.3 B) — not reproducible on the 96 GB card; document this in the report. On the 96 GB card, the ≤2^32 no-regression check above is the gate for this task.

- [ ] **Step 6: Commit**

```bash
git add lib/include/private/mimir/raytracing.hpp lib/src/raytracing.cpp lib/src/engine.cpp
git commit -m "feat(64bit): 64-bit PT particle_count + BLAS chunk index math"
```

---

## Task 5: Compute grid-stride (AABB writer + LOD scatter) over 64-bit counts

**Deliverable:** the AABB writer and the LOD scatter process an arbitrary 64-bit particle count via a bounded dispatch + 64-bit grid-stride loop, so PT and LOD reductions work past 2^32. Counts ≤ 2^32-1 are visited exactly once (identical result).

**Files:**
- Modify: `shaders/pathtrace_aabbs.slang`, `shaders/pathtrace_lod_scatter.slang`
- Modify: `lib/src/raytracing.cpp` (`AabbWriterPush`, writer dispatch), `lib/src/lod.cpp` (`LodScatterPush`, scatter dispatch, `recordReduction`/`init` count params)
- Modify: `lib/include/private/mimir/lod.hpp` (`recordReduction`/`init` particle-count params), `lib/src/engine.cpp` (drop the Task 1 LOD cast)

**Interfaces:**
- Consumes: `RayTracingContext::particle_count` is `uint64_t` (Task 4); `View::element_count` (Task 1).
- Produces: `LodContext::init(..., uint64_t particle_count)`, `LodContext::recordReduction(cmd, addr, uint64_t particle_count, slot)`; both compute shaders take a `uint64_t count` + `uint stride`.

- [ ] **Step 1: Grid-stride the AABB writer shader**

In `shaders/pathtrace_aabbs.slang`, change the push struct and the entry point:
```cpp
struct PushConstants
{
    Aabb* aabbs;
    float* positions;
    uint64_t count;   // 64-bit particle count
    float radius;
    uint stride;      // total dispatched threads (grid-stride step); <= 2^31
};
[[vk::push_constant]] PushConstants pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void writeAabbsMain(uint3 tid : SV_DispatchThreadID)
{
    for (uint64_t i = tid.x; i < pc.count; i += uint64_t(pc.stride))
    {
        uint64_t pos_addr = uint64_t(pc.positions) + i * 12ull;
        float px = *(float*)(pos_addr + 0);
        float py = *(float*)(pos_addr + 4);
        float pz = *(float*)(pos_addr + 8);
        float r  = pc.radius;
        Aabb a;
        a.minx = px - r; a.miny = py - r; a.minz = pz - r;
        a.maxx = px + r; a.maxy = py + r; a.maxz = pz + r;
        *(Aabb*)(uint64_t(pc.aabbs) + i * 24ull) = a;
    }
}
```

- [ ] **Step 2: Match `AabbWriterPush` and the writer dispatch**

In `lib/src/raytracing.cpp`, change `AabbWriterPush`:
```cpp
struct AabbWriterPush
{
    VkDeviceAddress aabbs;
    VkDeviceAddress positions;
    uint64_t count;
    float radius;
    uint32_t stride;
};
```
There are **two** `AabbWriterPush` fill + dispatch sites, and BOTH must adopt the new layout (a stale site would push the old struct and mismatch the widened shader): (a) the non-LOD writer (~1425-1434, count = `particle_count`), and (b) the LOD-path writer in `recordLodUpdate` (~1555, count = the `occupied` cell count, which is ≤ N^3 < 2^32 but still needs the `stride` field set). Use this pattern at each, substituting the appropriate count (`particle_count` vs `occupied`):
```cpp
    const uint32_t kMaxGroups = 1u << 25;   // 2^25 groups * 64 = 2^31 threads max (fits uint32 stride)
    uint32_t groups = (uint32_t)std::min<uint64_t>((COUNT + 63) / 64, kMaxGroups);
    if (groups == 0) groups = 1;
    AabbWriterPush push{
        .aabbs = /*...*/, .positions = /*...*/,
        .count = (uint64_t)COUNT, .radius = /*...*/, .stride = groups * 64u,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, iw_pipeline);
    vkCmdPushConstants(cmd, iw_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, groups, 1, 1);
```
At site (a) `COUNT = particle_count` (remove the Task 4 `(uint32_t)std::min` cast). At site (b) `COUNT = occupied` (keep the existing `.aabbs`/`.positions`/`.radius` values that site already uses).

- [ ] **Step 3: Grid-stride the LOD scatter shader**

In `shaders/pathtrace_lod_scatter.slang`, change the push struct's `uint count;` to `uint64_t count;`, add `uint stride;` at the end, and wrap the body in a grid-stride loop. The current body starts `uint i = tid.x; if (i >= pc.count) return;` — replace with:
```cpp
    for (uint64_t i = tid.x; i < pc.count; i += uint64_t(pc.stride))
    {
        uint64_t pos_addr = uint64_t(pc.positions) + i * 12ull;
        // ... existing per-particle body unchanged (uses i only for pos_addr; cell math is unchanged) ...
    }
```
Keep every existing line inside the loop (position read, cell index, `InterlockedAdd` count, the centroid sum block). Only the loop header and the `count`/`stride` push fields change.

- [ ] **Step 4: Match `LodScatterPush` + scatter dispatch + widen the LOD count params**

In `lib/src/lod.cpp`, change `LodScatterPush`:
```cpp
struct LodScatterPush
{
    VkDeviceAddress positions; VkDeviceAddress cellCounts; VkDeviceAddress cellSums;
    uint64_t count; uint32_t gridN; uint32_t centroid; uint32_t stride;
};
```
Change `LodContext::recordReduction(..., uint32_t particle_count, ...)` to `uint64_t particle_count` (and `LodContext::init(..., uint32_t particle_count)` to `uint64_t`) in both `lib/include/private/mimir/lod.hpp` and `lib/src/lod.cpp`. In `init`, `max_cells = std::min<uint64_t>(num_cells, particle_count)` is already 64-bit-safe. At the scatter dispatch (~263-267):
```cpp
    const uint32_t kMaxGroups = 1u << 25;
    uint32_t groups = (uint32_t)std::min<uint64_t>((particle_count + 63) / 64, kMaxGroups);
    if (groups == 0) groups = 1;
    LodScatterPush sp{ .positions = positions_addr, .cellCounts = cellcount_buffer.address,
        .cellSums = cellsum_addr, .count = particle_count, .gridN = grid_n,
        .centroid = centroid_flag, .stride = groups * 64u };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, scatter_pipeline);
    vkCmdPushConstants(cmd, scatter_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
    vkCmdDispatch(cmd, groups, 1, 1);
```
The emit dispatch (`(num_cells + 63) / 64`, ~280) and finalize (`1,1,1`) stay 1-D (N^3 < 2^32). In `lib/src/engine.cpp` (~271, and the PT `recordLodUpdate` path in raytracing.cpp if it passes `particle_count`), drop the Task 1 `(uint32_t)std::min` casts and pass the 64-bit `element_count`/`particle_count` directly.

- [ ] **Step 5: Build**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering`
Expected: compiles. The shaders are compiled at runtime (first use — Step 6 exercises them). The C++ push struct size must equal what the shader's `PushConstants` block expects. Add a `static_assert` next to each C++ struct with the size the compiler actually reports (do not guess — print `sizeof` once and pin it): `AabbWriterPush` is `2*8 + 8 + 4 + 4 = 32`; `LodScatterPush` is `3*8 + 8 + 4 + 4 + 4 = 44`, which the compiler pads to `48` (8-byte alignment from the `uint64_t` member) — so `static_assert(sizeof(LodScatterPush) == 48)`. `vkCmdPushConstants` pushes `sizeof(sp)`, and the Slang side must lay the block out to match; if Step 6 shows corruption or a validation size error, reconcile the field order/padding so the C++ `sizeof` and the shader block agree.

- [ ] **Step 6: Verify LOD scatter grid-stride ≤2^32 identical + >2^32 works**

Regression: `samples/remote-rendering/build/rr-server 9140 512 512 $((2**20)) 0 --light-model path-tracing --lod 32 2>&1 | grep -iE "LOD emitted|occupied"`
Expected: `LOD emitted 1472 occupied cells` — **identical** to Task 1's number (the grid-stride visits each particle once; determinism preserved).
>2^32 with LOD on the 96 GB card (positions ~60 GB, BVH over occupied cells is tiny): `samples/remote-rendering/build/rr-server 9141 512 512 5000000000 0 --light-model path-tracing --lod 128 2>&1 | grep -iE "LOD emitted|occupied|particles|reduction"`
Expected: logs `5000000000 particles` (no wrap), emits a plausible occupied-cell count (≤ 128^3), no crash — this exercises the scatter grid-stride over > 2^32 particles. Run it twice; the occupied count must be identical (determinism).

- [ ] **Step 7: Commit**

```bash
git add shaders/pathtrace_aabbs.slang shaders/pathtrace_lod_scatter.slang lib/src/raytracing.cpp lib/src/lod.cpp lib/include/private/mimir/lod.hpp lib/src/engine.cpp
git commit -m "feat(64bit): grid-stride AABB writer + LOD scatter over 64-bit counts"
```

---

## Task 6: Documentation

**Deliverable:** the sample docs and CLI help state that there is no particle-count cap (memory-bound), with the per-card ceiling guidance.

**Files:**
- Modify: `samples/remote-rendering/README.md`, `samples/remote-rendering/rr-server.cu` (`--help`/usage text), `samples/particles-kmodal-3d/benchmark_mimir.cpp` (usage text if it documents point_count)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the point_count docs**

In `samples/remote-rendering/README.md` and the rr-server `usage()` string, update the `point_count` description to: no fixed maximum — bounded by GPU memory (positions are 12 B/particle; path-tracing without `--lod` adds 24 B/particle for AABBs; `--lod` and raster keep it at ~12 B/particle). Note that an over-memory count is rejected up front. Add the ceiling guidance: e.g. "~7.5 B particles on a 96 GB GPU (none/phong/PT+LOD); ~2.6 B for path-tracing without LOD; more on larger-VRAM cards." Mirror the note in `benchmark_mimir.cpp`'s usage text if it lists positional `points`.

- [ ] **Step 2: Commit**

```bash
git add samples/remote-rendering/README.md samples/remote-rendering/rr-server.cu samples/particles-kmodal-3d/benchmark_mimir.cpp
git commit -m "docs(64bit): point_count is memory-bound, no fixed cap"
```

---

## Verification summary (what proves the feature)

- **≤2^32 unchanged:** Task 1/5 headless LOD count stays `1472`; Task 3 client render of a small scene is visually identical; Task 4 PT BLAS-chunk line unchanged at 2^28.
- **>2^32 renders:** Task 3 renders a seamless ~5 B point/phong/mesh cloud on the 96 GB card; Task 5 runs the LOD reduction over ~5 B particles (PT+LOD) with a correct, deterministic occupied count; both log the true count (no wrap).
- **Clean limits:** Task 2 rejects `count == 0` and an over-memory count up front (no `IOT`/OOM).
- **B300-only:** PT **without** LOD past 2^32 (154 GB AABBs at 4.3 B) is verified on the B300; the 96 GB card cannot hold it (documented).
