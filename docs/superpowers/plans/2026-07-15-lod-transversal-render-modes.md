# Transversal `--lod` Across Render Modes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--lod N` apply to every light model (`none`, `phong`, `phong-mesh`, `path-tracing`) by extracting LOD into one shared reduction stage that emits a reduced particle set, which raster (indirect instanced draw) and path-tracing (its existing AABB writer) both consume; and make `--size` a live multiplier under `--lod`.

**Architecture:** A new shared module (`lib/src/lod.cpp` + `lib/include/private/mimir/lod.hpp`) owns the accumulator/reduced/count buffers and the scatter/emit compute passes; its emit now writes occupied-cell **centroid positions** (not AABBs). Path-tracing runs its per-particle AABB writer over the reduced positions; raster binds the reduced positions as its vertex buffer and draws via `vkCmdDraw*Indirect` with the occupied count sourced from a GPU indirect-args buffer. `--lod 0` keeps every mode's current direct path.

**Tech Stack:** C++17, Vulkan 1.3 (indirect draw, buffer-device-address, int64 buffer atomics), Slang compute shaders (runtime-compiled), CUDA interop positions.

## Global Constraints

- One LOD codepath: the shared module in `lod.cpp/hpp`. No duplicated aggregation between raster and path-tracing.
- The reduced set is the single source of truth: reduced-position buffer (`min(N^3,P) * float3`, usage `VERTEX_BUFFER | SHADER_DEVICE_ADDRESS | STORAGE_BUFFER | TRANSFER_DST`) + a `uint` count buffer. The SAME reduced set feeds every consumer, so all four modes show the identical occupied count at a given N.
- Emit writes centroid POSITIONS (`float3`), appended via the `globalCount` atomic. Scatter is unchanged (count + int64 fixed-point centroid sum, `SCALE = 2^30`). Determinism preserved (integer atomics).
- BDA discipline unchanged: explicit 64-bit address arithmetic + cast, never pointer indexing.
- `--size` rule: lit modes (`phong`/`phong-mesh`/`path-tracing`) use `radius = cellFill * (default_size / LOD_REFERENCE_SIZE)`, `cellFill = coverage * cellSize / 2`, so the light model's default `--size` fills the cell; `none` uses `--size` as the pixel point size directly.
- `--lod 0` is byte-for-byte the current behavior in EVERY mode (no reduced buffer, no indirect draw). The cap is unchanged (VRAM-scaled, clamp N <= 1625).
- Path-tracing's occupied-cell count after the refactor MUST equal the shipped value: **13689** at `--lod 32` / 2^20. This is the checkpoint that the reduction result is unchanged by moving it out of `raytracing.cpp`.
- Build: `./mimir-build-from-change.sh` (lib) + `./samples-build-from-change.sh --sample remote-rendering` (relink). Clock-skew warnings are harmless. Release build → Vulkan validation OFF; use `SPDLOG_LEVEL=info` for init-time logs. Evidence is rendered output + logged counts + determinism.

### Reference run commands
Fast smoke (~2^20 particles). Swap `--light-model` to test each mode; connect `rr-client` to see the image (raster modes render fast, so a client shows them live):
```bash
samples/remote-rendering/build/rr-server 9000 1920 1080 $((2**20)) 413111 10000 \
  --pcolor 1.0,0.05,0.05 --background 0.2 --k 64 --epsilon 0.07 \
  --light-model <none|phong|phong-mesh|path-tracing> --size <s> --steps-per-frame 1 --fps 60 --fly --lod 32
```
Headless count check: the server logs the occupied-cell count without a client. The occupied count must be **13689** at `--lod 32` / 2^20 in every mode.

---

## File Structure

- Create: `lib/include/private/mimir/lod.hpp` — `LodContext` (buffers, pipelines, `recordReduction`, `readCount`, `sphereRadius(default_size)`, `reducedPositionsBuffer()`, `countBuffer()`).
- Create: `lib/src/lod.cpp` — the shared reduction stage (moved from raytracing.cpp) + emit-writes-positions + a finalize-indirect-args helper.
- Move-from/modify: `shaders/pathtrace_lod_scatter.slang` (unchanged logic; may relocate/rename under a neutral name), `shaders/pathtrace_lod_emit.slang` → emit writes positions.
- Create: `shaders/lod_indirect_args.slang` — 1-thread finalize writing a `VkDraw*IndirectCommand` from the count.
- Modify: `lib/src/raytracing.cpp` — drop internal scatter/emit; LOD path runs the AABB writer over the reduced positions from `LodContext`.
- Modify: `lib/src/engine.cpp` — own a `LodContext`; call `recordReduction` early each frame when LOD on; raster `drawElements` binds reduced vbo + indirect draw; `--size` multiplier.
- Modify: `samples/remote-rendering/README.md` — `--lod` now transversal + `--size` behavior.

---

## Task 1: Extract the shared LOD module; path-tracing consumes the reduced set (behavior-identical)

Move LOD out of `raytracing.cpp` into a shared `LodContext`; change emit to write centroid positions; make path-tracing run its existing AABB writer over the reduced positions. Net observable path-tracing behavior is IDENTICAL (same 13689 count, same image); `--size` becomes live for PT. No raster changes yet.

**Files:** Create `lib/include/private/mimir/lod.hpp`, `lib/src/lod.cpp`; modify `shaders/pathtrace_lod_emit.slang`, `lib/src/raytracing.cpp`, `lib/src/engine.cpp`, and the sample/lib CMake if a new .cpp needs listing (grep how `raytracing.cpp` is listed in `lib/CMakeLists.txt` and add `lod.cpp` the same way).

**Interfaces:**
- Produces: `class LodContext { void init(...); void recordReduction(VkCommandBuffer, VkDeviceAddress positions, uint32_t particle_count); uint32_t readCount(); VkBuffer reducedPositionsBuffer(); VkDeviceAddress reducedPositionsAddress(); float sphereRadius(float default_size) const; bool active() const; uint32_t cells() const; ... };` plus `static constexpr float LOD_REFERENCE_SIZE`.
- Consumes: `RayTracingContext::int64_atomics` (now passed to LodContext), the AABB writer (`iw_pipeline`).

- [ ] **Step 1: Create `LodContext` skeleton and move the buffers/pipelines**

Create `lod.hpp`/`lod.cpp`. Move from `raytracing.cpp` into `LodContext`: `lod_cellcount_buffer`, `lod_cellsum_buffer`, `lod_counter_buffer`, `LOD_FIXEDPOINT_SCALE`, `LOD_COVERAGE`, the scatter/emit compute pipelines + descriptor/push structs, and the `recordLodUpdate` aggregation body (clear→scatter→emit). Add a new **reduced-position buffer** `RtBuffer lod_reduced_pos` sized `min(N^3,P) * sizeof(float3)`, usage `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT`, DEVICE_LOCAL, wantAddress=true. `LodContext::init` takes device/mem-props/submit/`int64_atomics` and the grid N + particle count. Keep `makeBuffer`/`destroyBuffer` (share or duplicate the small helpers).

- [ ] **Step 2: Change emit to write centroid POSITIONS**

In `shaders/pathtrace_lod_emit.slang`, replace the AABB output with a `float3` reduced-position output (BDA pointer `reducedPos`). For an occupied cell: compute the centroid (same int64→double dequantize as now), `uint slot; InterlockedAdd(globalCount[0], 1u, slot);` then write the centroid to `*(float3*)(uint64_t(pc.reducedPos) + uint64_t(slot)*12ull) = center;`. Drop the `radius`/AABB fields from the emit push struct (radius is now a consumer concern). Keep the cell-center fallback (when `centroid==0`) writing the cell center as the position.

- [ ] **Step 3: Path-tracing consumes reduced positions**

In `raytracing.cpp`, `recordUpdateScene` LOD path becomes: `lod.recordReduction(cmd, position_address, particle_count); uint32_t occupied = lod.readCount(); occupied = min(occupied, lod_max_cells);` then run the EXISTING AABB writer (`iw_pipeline`) with `positions = lod.reducedPositionsAddress()`, `count = occupied`, `radius = lod.sphereRadius(particle_radius)`, then `recordBlasBuildChunks(..., override_prims=occupied)` + TLAS — exactly the current build tail. Remove the old in-`raytracing.cpp` scatter/emit and the AABB-writing emit. The AABB buffer is now written by the standard AABB writer over reduced positions (size it for `lod_max_cells` as today). Keep the emit→build barrier (now: reduction → AABB-writer → build; ensure the AABB-writer→build barrier and the reduction→AABB-writer barrier are both present).

- [ ] **Step 4: `--size` multiplier + engine owns the LodContext**

Add `LodContext` as an engine member; `init` it when `pt_lod_cells > 0`. Implement `sphereRadius(default_size) = cellFill * (default_size / LOD_REFERENCE_SIZE)` with `cellFill = LOD_COVERAGE * (2.0f/N) * 0.5f` and `LOD_REFERENCE_SIZE` = the sample's lit default size (grep the lit `--size` default in `rr-server.cu`, e.g. `size_px/100`; pick the value that makes the current default fill the cell — document it). Pass the RT context the LodContext (or have engine drive both). For this task only PT reads it; raster is unchanged.

- [ ] **Step 5: Build + verify PT is behavior-identical and `--size` is live**

Build both. Run PT `--lod 32` at 2^20: occupied count MUST be **13689** (unchanged), deterministic across two runs; image unchanged from before this task. Run PT `--lod 32 --size <2x default>` and confirm the AABB radius scales (blobs chunkier) — inspect via a client, or at minimum confirm `sphereRadius` returns the expected value in a log line. Run PT `--lod 0` unchanged.
If the count is not 13689, the reduction moved incorrectly — stop and diagnose.

- [ ] **Step 6: Commit**
```bash
git add lib/include/private/mimir/lod.hpp lib/src/lod.cpp shaders/pathtrace_lod_emit.slang \
        lib/src/raytracing.cpp lib/src/engine.cpp lib/CMakeLists.txt
git commit -m "feat(lod): extract shared LOD reduction module; PT consumes reduced positions"
```

---

## Task 2: Raster point modes (`none`, `phong`) via indirect draw

Give `none` and `phong` LOD by binding the reduced-position buffer as the marker vertex buffer and drawing with `vkCmdDrawIndirect`, count sourced from a GPU indirect-args buffer.

**Files:** Create `shaders/lod_indirect_args.slang`; modify `lib/src/lod.cpp`/`lod.hpp` (indirect-args buffer + finalize pipeline), `lib/src/engine.cpp` (drawElements + recordReduction call site + reduced vbo bind + `none` pixel size).

**Interfaces:**
- Consumes: `LodContext` reduced positions + count (Task 1).
- Produces: `LodContext::recordIndirectArgs(cmd, uint32_t fixed_field, uint32_t varying_offset)` writing a `VkDrawIndirectCommand`; `VkBuffer LodContext::indirectBuffer()`.

- [ ] **Step 1: Indirect-args finalize shader**

Create `shaders/lod_indirect_args.slang`: 1 thread, push constants `{ uint* indirect; uint* count; uint varyingByteOffset; uint fixedInstanceCount; }`. Pre-condition: host has zero/format-filled the indirect buffer. The shader reads `count[0]` and writes it into the indirect command's varying field (`*(uint*)(uint64_t(pc.indirect) + varyingByteOffset) = count[0]`), and writes `fixedInstanceCount` into the instanceCount field for point mode. Keep it generic (offsets passed in). Build a compute pipeline for it in `LodContext`.

- [ ] **Step 2: Indirect buffer + recordIndirectArgs in LodContext**

Allocate `lod_indirect_buffer` (`sizeof(VkDrawIndirectCommand)` = 16 B, usage `INDIRECT_BUFFER | STORAGE | TRANSFER_DST`, DEVICE_LOCAL, wantAddress). `recordIndirectArgs`: `vkCmdFillBuffer` the fixed template (firstVertex=0, firstInstance=0, instanceCount=1 for point), a barrier, dispatch the finalize (1 thread) writing vertexCount=count at offset 0, a barrier (`SHADER_WRITE → INDIRECT_COMMAND_READ`, stage `DRAW_INDIRECT`).

- [ ] **Step 3: engine drives reduction + indirect draw for point modes**

In `renderFrame`/`drawElements`: when `lod.active()` and the view is a point-mode marker (render_mode Flat2D or Sphere3D, `!use_ibo`): call `lod.recordReduction(cmd, ...)` + `lod.recordIndirectArgs(cmd, /*fixed instanceCount*/1, /*vertexCount offset*/0)` early in the frame cmd; bind `lod.reducedPositionsBuffer()` as vbo 0 instead of the interop buffer; replace `vkCmdDraw(...)` with `vkCmdDrawIndirect(cmd, lod.indirectBuffer(), 0, 1, 0)`. For `none`, keep the marker pixel size = `--size` (no cell-fill); for `phong`, set the lit marker size (`engine.cpp:2232`) to `lod.sphereRadius(default_size)`. When `!lod.active()`, the existing direct path is unchanged. (recordReduction must run once per frame even with multiple views; guard so it isn't dispatched per-view — reduce once, reuse.)

- [ ] **Step 4: Build + verify none/phong**

Build. With a client: `--light-model none --lod 32` shows a reduced point cloud; `--light-model phong --lod 32` shows reduced lit spheres; sweeping `--size` in phong changes blob size; `--lod 0` shows the full cloud in each. Headless: the occupied count logs **13689** at --lod 32 (same as PT). Determinism across runs.

- [ ] **Step 5: Commit**
```bash
git add shaders/lod_indirect_args.slang lib/src/lod.cpp lib/include/private/mimir/lod.hpp lib/src/engine.cpp
git commit -m "feat(lod): none/phong raster LOD via indirect draw over reduced positions"
```

---

## Task 3: `phong-mesh` via indexed indirect draw

Give `phong-mesh` LOD: instanceCount (one icosphere per occupied cell) is dynamic; use `vkCmdDrawIndexedIndirect`.

**Files:** modify `lib/src/lod.cpp`/`lod.hpp` (support the indexed-indirect command layout), `lib/src/engine.cpp` (mesh draw path).

**Interfaces:**
- Consumes: Task 2's `recordIndirectArgs`/indirect buffer, generalized for `VkDrawIndexedIndirectCommand`.

- [ ] **Step 1: Support the indexed-indirect command**

`VkDrawIndexedIndirectCommand{ indexCount, instanceCount, firstIndex, vertexOffset, firstInstance }` (20 B). For mesh the VARYING field is `instanceCount` (byte offset 4); the FIXED field is `indexCount` (= icosphere index count, offset 0). Size `lod_indirect_buffer` to `max(sizeof(VkDrawIndirectCommand), sizeof(VkDrawIndexedIndirectCommand))`. Generalize `recordIndirectArgs` to take the command's fixed field + value and the varying-field byte offset (indexed: fill indexCount, write instanceCount=count at offset 4). The finalize shader already writes `count` at a given offset — pass offset 4 and pre-fill indexCount.

- [ ] **Step 2: engine mesh draw path**

In `drawElements` for a mesh marker (`use_ibo`, render_mode SphereMesh) when `lod.active()`: bind the reduced positions as the per-INSTANCE vertex attribute (the buffer that currently supplies per-instance particle positions — confirm which vbo slot holds instance positions vs the template icosphere vertices; the reduced buffer replaces the instance-position vbo, NOT the icosphere template). Call `lod.recordReduction` + `lod.recordIndirectArgs(cmd, /*fixed indexCount*/ sphere_index_count, /*varying instanceCount offset*/4)`; replace `vkCmdDrawIndexed(...)` with `vkCmdDrawIndexedIndirect(cmd, lod.indirectBuffer(), 0, 1, 0)`. Set the mesh marker size to `lod.sphereRadius(default_size)`. `!lod.active()` path unchanged.

- [ ] **Step 3: Build + verify phong-mesh**

Build. With a client: `--light-model phong-mesh --lod 32` shows reduced icospheres (fewer, cell-sized); `--size` scales them; `--lod 0` shows full-resolution icospheres. Headless count logs **13689** at --lod 32. Determinism.

- [ ] **Step 4: Commit**
```bash
git add lib/src/lod.cpp lib/include/private/mimir/lod.hpp lib/src/engine.cpp
git commit -m "feat(lod): phong-mesh LOD via indexed indirect draw"
```

---

## Task 4: Documentation

**Files:** modify `samples/remote-rendering/README.md`.

- [ ] **Step 1: Update `--lod` docs**

Amend the `--lod` section: it now applies to ALL light models (`none`, `phong`, `phong-mesh`, `path-tracing`), reducing how many primitives are drawn regardless of shading. Document `--size` behavior under `--lod`: lit modes scale the cell-fill radius by `--size` (default fills the cell); `none` uses `--size` as the pixel point size. Note the occupied count is identical across modes at a given N. Keep the determinism / VRAM-cap notes.

- [ ] **Step 2: Commit**
```bash
git add samples/remote-rendering/README.md
git commit -m "docs(lod): --lod now transversal across all light models"
```

---

## Self-Review Notes

- **Spec coverage:** shared module extraction (Task 1), emit-writes-positions (Task 1 Step 2), PT consumes reduced set behavior-identically (Task 1 Steps 3/5, 13689 checkpoint), `--size` multiplier lit + none pixel size (Task 1 Step 4, Task 2 Step 3, Task 3 Step 2), indirect draw for none/phong (Task 2) and phong-mesh (Task 3), determinism + `--lod 0` no-op per mode (each task's verify), cap unchanged (untouched), docs (Task 4).
- **Placeholder scan:** the mechanical code moves are described precisely (which symbols move where) rather than reproduced verbatim — appropriate for a refactor, not a placeholder. The novel code (emit position write, finalize shader, indirect draw calls, `--size` formula) is concrete. One value to pin during Task 1: `LOD_REFERENCE_SIZE` (the lit default `--size` that makes cell-fill = 1.0) — resolve from `rr-server.cu`'s default and document it.
- **Type consistency:** `LodContext`, `recordReduction`, `readCount`, `reducedPositionsBuffer/Address`, `sphereRadius`, `recordIndirectArgs`, `indirectBuffer`, `LOD_REFERENCE_SIZE`, `lod_reduced_pos`/`lod_indirect_buffer` are named consistently across Tasks 1-3.
- **Risks to watch:** (a) Task 1 is the big refactor — the 13689 checkpoint is the guardrail that the reduction result is unchanged; (b) `recordReduction` must run ONCE per frame, not per view (guard against multi-view double-dispatch); (c) for phong-mesh confirm which vbo slot is the per-instance position (vs the icosphere template) before swapping it to the reduced buffer; (d) the reduced buffer is single-buffered + barrier-serialized (v1) — watch for a missing barrier between the reduction writes and the draw/AABB-writer reads in each mode; (e) `--lod 0` must bypass reduction, reduced-vbo bind, and indirect draw entirely in every mode.
