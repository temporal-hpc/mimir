# LOD Voxel Rendering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--lod-voxel` mode that renders path-traced LOD representatives as solid, grid-aligned, cell-filling cubes (voxels) instead of inscribed spheres.

**Architecture:** The AABB writer already emits a cube AABB per representative and the intersection shader already has a box/slab branch (`pathtrace.slang:308`) selected by `sun_dir.w >= 0.5`. We add a `lod_voxel` flag that (a) makes the LOD path select that box branch (opaque), and (b) overrides the representative radius to half the cell edge so cubes tile. The `voxel_boxes` CA3D pipeline is deliberately left untouched — `lod_voxel` only affects `sun_dir.w` and the LOD radius.

**Tech Stack:** C++17, CUDA, Vulkan (hardware ray tracing), Slang shaders. Build via CMake (two trees: `build/` for `libmimir.a`, `samples/remote-rendering/build/` for `rr-server`/`rr-client`).

## Global Constraints

- **No shader changes.** The box branch (`shaders/pathtrace.slang:287-360`) already exists; do not edit any `.slang` file.
- **Default off / no behavior change when unset.** `--lod-voxel` absent ⇒ spheres, byte-identical to today.
- **Do not touch the `voxel_boxes` CA3D pipeline.** `lod_voxel` must never enter the branches at `raytracing.cpp:1349`, `:1370`, `:1388`. It only affects `sun_dir.w` (`engine.cpp:2537`) and the AABB radius (`raytracing.cpp:1621`).
- **Fixed domain.** The LOD grid spans `[-1,1]^3`; cell edge `e = 2 / grid_n`; full-cell voxel half-extent `= e/2 = 1 / grid_n`.
- **Opaque only.** LOD voxels use opacity `1.0` ⇒ `sun_dir.w = 2.0`. No opacity flag.
- **Build order:** after editing `lib/` files, rebuild `libmimir.a` first, then relink `rr-server` (the sample links the static lib):
  ```bash
  touch lib/src/remote.cpp   # only if clock-skew warnings cause skipped rebuilds
  cmake --build build --target mimir -j
  cmake --build samples/remote-rendering/build --target rr-server -j
  ```
- **Verification is integration-style** (headless run + log assertions + saved `.ppm`): this codebase has no unit-test harness for the render pipeline. Mirror the render_ms-fix verification: background the server, run `rr-client` headless, grep the server log.

---

## File Structure

- `lib/include/public/mimir/options.hpp` — add `bool pt_lod_voxel` to `ViewerOptions` (public knob).
- `lib/include/private/mimir/raytracing.hpp` — add `bool lod_voxel` to `RayTracingContext` (read by the radius override and the engine's `shape_w`).
- `lib/include/private/mimir/lod.hpp` + `lib/src/lod.cpp` — add `LodContext::voxelHalfExtent()` (the `1/grid_n` cell half-edge; keeps domain math in one place).
- `lib/src/engine.cpp` — set `raytracing.lod_voxel` from options; OR it into the `shape_w` selector.
- `lib/src/raytracing.cpp` — pick voxel half-extent vs sphere radius in the LOD AABB writer; update the emit log.
- `samples/remote-rendering/rr-server.cu` — parse `--lod-voxel`, help text, force cell-center + `--size`-ignored notes, assign into options.

---

## Task 1: `--lod-voxel` flag plumbed end-to-end (placement forced, no render change yet)

Adds the public option, the private raytracing field, the CLI flag, and the cell-center force + notes. After this task the flag is recognized and forces cell-center placement, but rendering is still spheres (shape/radius wired in Task 2). This is independently reviewable: a reviewer verifies the flag parses, plumbs, forces placement, and logs — without any visual change.

**Files:**
- Modify: `lib/include/public/mimir/options.hpp:227` (after `bool lod_centroid = true;`)
- Modify: `lib/include/private/mimir/raytracing.hpp:155` (near `uint32_t lod_cells = 0;`)
- Modify: `lib/src/engine.cpp:237` (next to `raytracing.lod_cells = options.pt_lod_cells;`)
- Modify: `samples/remote-rendering/rr-server.cu` (declarations ~222, parse ~256, help ~96, force+assign ~417)

**Interfaces:**
- Produces: `ViewerOptions::pt_lod_voxel` (bool, default `false`); `RayTracingContext::lod_voxel` (bool, default `false`). Task 2 consumes both.

- [ ] **Step 1: Add the public option field**

In `lib/include/public/mimir/options.hpp`, immediately after the `bool lod_centroid = true;` line (currently `:227`), add:

```cpp
    // Render LOD representatives as solid grid-aligned cubes (voxels) instead of inscribed
    // spheres. Opt-in visual identity for the reduced level of detail (only meaningful when
    // pt_lod_cells > 0 under path tracing). Forces cell-center placement and full-cell fill;
    // see rr-server --lod-voxel. Independent of the CA3D voxel_boxes pipeline.
    bool pt_lod_voxel = false;
```

- [ ] **Step 2: Add the private raytracing field**

In `lib/include/private/mimir/raytracing.hpp`, immediately after `uint32_t lod_cells = 0;` (currently `:155`), add:

```cpp
    // When true, the LOD AABB writer sizes each representative to a full cell cube (voxelHalfExtent)
    // and the engine selects the box branch of the intersection shader (sun_dir.w >= 0.5, opaque).
    // Affects ONLY the shape + radius, never the LOD reduction pipeline. Set from ViewerOptions::pt_lod_voxel.
    bool lod_voxel = false;
```

- [ ] **Step 3: Plumb options → raytracing in the engine**

In `lib/src/engine.cpp`, immediately after `raytracing.lod_cells = options.pt_lod_cells;` (currently `:237`), add:

```cpp
                raytracing.lod_voxel = options.pt_lod_voxel;
```

- [ ] **Step 4: Add the rr-server flag declarations**

In `samples/remote-rendering/rr-server.cu`, next to the LOD option declarations (near `bool lod_centroid = true;` at `:222`), add:

```cpp
    bool lod_voxel_render   = false;  // --lod-voxel: draw LOD reps as full-cell cubes (forces cell-center)
    bool size_set           = false;  // whether --size was explicitly passed (for the voxel-mode ignore note)
```

- [ ] **Step 5: Parse `--lod-voxel` and record `--size` being set**

In `samples/remote-rendering/rr-server.cu`, change the `--size` parse (currently `:247`) to record it, and add the `--lod-voxel` parse after the `--lod-placement` branch (currently `:256`).

Change:
```cpp
            else if (a == "--size")        size_px = std::stof(v);
```
to:
```cpp
            else if (a == "--size")      { size_px = std::stof(v); size_set = true; }
```

After the `--lod-placement` line, add (note: `--lod-voxel` is a boolean flag with no value argument — insert it in the boolean-flag group, not the value-consuming group; match the file's existing boolean-flag parse style):

```cpp
            else if (a == "--lod-voxel")   lod_voxel_render = true;
```

- [ ] **Step 6: Add help text**

In `samples/remote-rendering/rr-server.cu`, in the usage string after the `--lod-placement` block (the line ending at `:96`, before `--sort-every` at `:97`), add:

```cpp
        "  --lod-voxel        Render each LOD representative as a solid grid-aligned cube (voxel)\n"
        "                     instead of a sphere -- a distinct visual identity for the reduced\n"
        "                     level of detail. Forces cell-center placement and full-cell fill\n"
        "                     (ignores --size for the box extent). Default: off (spheres).\n"
```

- [ ] **Step 7: Force cell-center + emit notes, then assign into options**

In `samples/remote-rendering/rr-server.cu`, immediately before `options.pt_lod_cells = lod_cells;` (currently `:417`), add:

```cpp
    // LOD voxel mode: full-cell cubes only tile on the grid lattice, so force cell-center placement
    // (centroid sits off-lattice and breaks tiling) and ignore --size (the box always fills the cell).
    if (lod_voxel_render && lod_cells > 0)
    {
        if (lod_centroid)
        {
            fprintf(stdout, "rr-server: --lod-voxel forces cell-center placement "
                            "(centroid is off-lattice and breaks tiling)\n");
            lod_centroid = false;
        }
        if (size_set)
        {
            fprintf(stdout, "rr-server: --lod-voxel ignores --size; voxels always fill the cell\n");
        }
    }
```

Then, immediately after `options.lod_centroid = lod_centroid;` (currently `:418`), add:

```cpp
    options.pt_lod_voxel      = lod_voxel_render;
```

- [ ] **Step 8: Build the library and relink the server**

Run:
```bash
touch lib/src/remote.cpp
cmake --build build --target mimir -j
cmake --build samples/remote-rendering/build --target rr-server -j
```
Expected: both targets build; `rr-server` relinks (ignore "clock skew" warnings — the `touch` + full rebuild covers them).

- [ ] **Step 9: Verify the flag is recognized and forces cell-center**

Start the server headless in the background with `--lod-placement centroid --lod-voxel` and capture its log:
```bash
SP=/tmp/lodvoxel-t1.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 200 \
  --light-model path-tracing --spp 1 --fps 60 --lod 64 --lod-placement centroid --lod-voxel \
  > "$SP" 2>&1 &
echo $! > /tmp/lodvoxel-t1.pid
# wait for bind
for i in $(seq 1 30); do grep -q "listening on port" "$SP" && break; sleep 1; done
grep -q "forces cell-center placement" "$SP" && echo "PASS: placement forced" || echo "FAIL: no force note"
```
Expected: `PASS: placement forced` (the `rr-server: --lod-voxel forces cell-center placement ...` line is present).

- [ ] **Step 10: Verify default-off is unchanged (regression) and stop the server**

```bash
kill "$(cat /tmp/lodvoxel-t1.pid)" 2>/dev/null; sleep 1
SP2=/tmp/lodvoxel-t1-off.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 200 \
  --light-model path-tracing --spp 1 --fps 60 --lod 64 --lod-placement centroid \
  > "$SP2" 2>&1 &
echo $! > /tmp/lodvoxel-t1off.pid
for i in $(seq 1 30); do grep -q "listening on port" "$SP2" && break; sleep 1; done
grep -q "forces cell-center placement" "$SP2" && echo "FAIL: fired without flag" || echo "PASS: no force note when flag absent"
kill "$(cat /tmp/lodvoxel-t1off.pid)" 2>/dev/null
```
Expected: `PASS: no force note when flag absent`.

- [ ] **Step 11: Commit**

```bash
git add lib/include/public/mimir/options.hpp lib/include/private/mimir/raytracing.hpp \
        lib/src/engine.cpp samples/remote-rendering/rr-server.cu
git commit -m "feat(rr): add --lod-voxel flag (plumbing + force cell-center placement)

Recognizes --lod-voxel end-to-end (ViewerOptions::pt_lod_voxel ->
RayTracingContext::lod_voxel), forces cell-center placement (centroid is
off-lattice and breaks full-cell tiling), and notes that --size is ignored
for voxel extent. No render change yet (shape/radius wired next).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W"
```

---

## Task 2: Box shape + full-cell radius (the actual voxel rendering)

Wires the flag into the intersection-shader shape selector and the AABB radius so LOD reps render as opaque, cell-filling, grid-aligned cubes. Independently reviewable: a reviewer confirms the emit log reports voxel mode with `half-extent = 1/grid_n`, the saved frame shows cubes, and a run without the flag still shows spheres.

**Files:**
- Modify: `lib/include/private/mimir/lod.hpp:144` (declare `voxelHalfExtent`)
- Modify: `lib/src/lod.cpp:470` (define `voxelHalfExtent`, after `sphereRadius`)
- Modify: `lib/src/raytracing.cpp:1621` (radius select) and `:1673-1676` (emit log)
- Modify: `lib/src/engine.cpp:2537` (shape_w selector)

**Interfaces:**
- Consumes: `RayTracingContext::lod_voxel` (Task 1); `ViewerOptions::pt_lod_voxel` (Task 1, though the engine reads it via the `raytracing.lod_voxel` field it already set).
- Produces: `float LodContext::voxelHalfExtent() const` — returns `1.0f / float(grid_n)` (half the `[-1,1]^3` cell edge). Used by the AABB writer and the emit log.

- [ ] **Step 1: Declare `voxelHalfExtent` on LodContext**

In `lib/include/private/mimir/lod.hpp`, immediately after the `float sphereRadius(float default_size) const;` declaration (currently `:144`), add:

```cpp
    // Half the LOD cell edge over the fixed [-1,1]^3 domain: e/2 = (2/grid_n)/2 = 1/grid_n. Used as
    // the AABB half-extent for --lod-voxel so cell-center-placed cubes tile face-to-face (no --size
    // scaling, no LOD_COVERAGE overflow -- exact cell fill).
    float voxelHalfExtent() const;
```

- [ ] **Step 2: Define `voxelHalfExtent`**

In `lib/src/lod.cpp`, immediately after the closing brace of `sphereRadius` (currently `:470`), add:

```cpp
float LodContext::voxelHalfExtent() const
{
    // Full-cell cube: half-extent = half the cell edge. Domain is [-1,1]^3 (edge = 2/grid_n), so
    // half-extent = 1/grid_n. grid_n > 0 whenever active() (guarded by the caller: lod != nullptr).
    return 1.0f / float(grid_n);
}
```

- [ ] **Step 3: Select voxel half-extent vs sphere radius in the AABB writer**

In `lib/src/raytracing.cpp`, in the LOD AABB-writer push-constant (currently `:1621`), change:

```cpp
        .count = (uint64_t)occupied, .radius = lod->sphereRadius(particle_radius), .stride = groups * 64u,
```
to:
```cpp
        .count = (uint64_t)occupied,
        .radius = lod_voxel ? lod->voxelHalfExtent() : lod->sphereRadius(particle_radius),
        .stride = groups * 64u,
```

- [ ] **Step 4: Report voxel mode in the emit log**

In `lib/src/raytracing.cpp`, replace the emit-log `spdlog::info(...)` inside `if (first_build)` (currently `:1673-1676`) with a branch on `lod_voxel`:

```cpp
        if (lod_voxel)
        {
            spdlog::info("Path tracing: LOD emitted {} occupied cells (reduction {:.0f}:1 vs {} particles); "
                "voxel half-extent {:.5f} (full-cell cubes, cell edge {:.5f})",
                occupied, occupied ? double(particle_count) / double(occupied) : 0.0, particle_count,
                lod->voxelHalfExtent(), 2.0f * lod->voxelHalfExtent());
        }
        else
        {
            spdlog::info("Path tracing: LOD emitted {} occupied cells (reduction {:.0f}:1 vs {} particles); "
                "sphere radius {:.5f} (--size {:.5f}, cell-fill at {:.5f})",
                occupied, occupied ? double(particle_count) / double(occupied) : 0.0, particle_count,
                lod->sphereRadius(particle_radius), particle_radius, LodContext::LOD_REFERENCE_SIZE);
        }
```

- [ ] **Step 5: OR `lod_voxel` into the shape selector**

In `lib/src/engine.cpp`, replace the `shape_w` line (currently `:2537`):

```cpp
        float shape_w = raytracing.voxel_boxes ? (1.f + raytracing.voxel_opacity) : 0.f;
```
with:
```cpp
        // voxel_boxes = the CA3D voxel pipeline; lod_voxel = LOD reps drawn as opaque cubes. Either
        // one selects the box branch of the intersection shader; opacity is voxel_opacity for CA3D
        // and 1.0 (opaque) for LOD voxels. The two flags are mutually exclusive in practice.
        const float box_opacity = raytracing.voxel_boxes ? raytracing.voxel_opacity : 1.f;
        float shape_w = (raytracing.voxel_boxes || raytracing.lod_voxel) ? (1.f + box_opacity) : 0.f;
```

- [ ] **Step 6: Build the library and relink the server**

Run:
```bash
touch lib/src/remote.cpp
cmake --build build --target mimir -j
cmake --build samples/remote-rendering/build --target rr-server -j
```
Expected: both build; `rr-server` relinks.

- [ ] **Step 7: Verify the emit log reports voxel half-extent = 1/grid_n**

Start the server with `--lod 64 --lod-voxel` and check the emit log. For `grid_n = 64`, `voxelHalfExtent = 1/64 = 0.01563` and `cell edge = 0.03125`:
```bash
SP=/tmp/lodvoxel-t2.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 200 \
  --light-model path-tracing --spp 1 --fps 60 --lod 64 --lod-voxel \
  > "$SP" 2>&1 &
echo $! > /tmp/lodvoxel-t2.pid
for i in $(seq 1 30); do grep -q "listening on port" "$SP" && break; sleep 1; done
grep -E "voxel half-extent 0\.01563 \(full-cell cubes, cell edge 0\.03125\)" "$SP" \
  && echo "PASS: voxel emit log correct" || { echo "FAIL: emit log"; grep "LOD emitted" "$SP"; }
```
Expected: `PASS: voxel emit log correct`.

- [ ] **Step 8: Verify cubes render (headless client → PPM) and render_ms is not inflated**

With the Task-2 server still running, drive a bounded headless client, then inspect:
```bash
timeout 60 samples/remote-rendering/build/rr-client 127.0.0.1 9000 "" tcp 40 2>&1 | tail -3
# server-side render split for a voxel frame:
grep "ms render" "$SP" | tail -2
cp rr-client.ppm /tmp/lodvoxel-voxel.ppm
kill "$(cat /tmp/lodvoxel-t2.pid)" 2>/dev/null
```
Expected: client prints `received 40 frames` and `saved rr-client.ppm`; the server `ms render` line is a small single-digit-ms trace (not 0.00, not inflated) with `lod ... ms` broken out separately. Open `/tmp/lodvoxel-voxel.ppm` and confirm the reduced cloud is made of **cubes** (flat-shaded square faces / contiguous blocks), not round blobs.

- [ ] **Step 9: Regression — without the flag, still spheres**

```bash
SP3=/tmp/lodvoxel-t2-sphere.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 200 \
  --light-model path-tracing --spp 1 --fps 60 --lod 64 \
  > "$SP3" 2>&1 &
echo $! > /tmp/lodvoxel-t2s.pid
for i in $(seq 1 30); do grep -q "listening on port" "$SP3" && break; sleep 1; done
grep -q "sphere radius" "$SP3" && echo "PASS: sphere path intact" || echo "FAIL: sphere log missing"
timeout 60 samples/remote-rendering/build/rr-client 127.0.0.1 9000 "" tcp 40 2>&1 | tail -1
cp rr-client.ppm /tmp/lodvoxel-sphere.ppm
kill "$(cat /tmp/lodvoxel-t2s.pid)" 2>/dev/null
```
Expected: `PASS: sphere path intact`; `/tmp/lodvoxel-sphere.ppm` shows round blobs. Compare against `/tmp/lodvoxel-voxel.ppm` to confirm the visual difference.

- [ ] **Step 10: Commit**

```bash
git add lib/include/private/mimir/lod.hpp lib/src/lod.cpp lib/src/raytracing.cpp lib/src/engine.cpp
git commit -m "feat(rr): render LOD representatives as full-cell voxels under --lod-voxel

Selects the existing box branch of the intersection shader (sun_dir.w=2.0,
opaque) for the LOD path and sizes each representative AABB to half the cell
edge (LodContext::voxelHalfExtent = 1/grid_n) so cell-center-placed cubes
tile face-to-face. Emit log reports voxel mode. No shader changes; the CA3D
voxel_boxes pipeline is untouched.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W"
```

---

## Self-Review

**Spec coverage:**
- Goal (voxel visual identity) → Task 2 (shape + radius). ✓
- `voxel_boxes` vs `lod_voxel` separation → Task 1 Step 2/3 (separate field), Task 2 Step 5 (OR only in `shape_w`), Global Constraints. ✓
- Full-cell tiling radius `e/2 = 1/grid_n` → Task 2 Steps 1-3 (`voxelHalfExtent`). ✓
- `--size` ignored + note → Task 1 Step 5/7. ✓
- Auto-force cell-center + note → Task 1 Step 7. ✓
- Opaque only (`shape_w = 2.0`) → Task 2 Step 5 (`box_opacity = 1.f`). ✓
- CLI `--lod-voxel` default off → Task 1 Steps 4-7. ✓
- No shader changes → Global Constraints; no `.slang` file in any task. ✓
- Testing (headless run, log asserts, PPM, regression) → Task 1 Steps 9-10, Task 2 Steps 7-9. ✓
- Non-goals (translucency, triangle cubes, reduction changes) → none introduced. ✓

**Placeholder scan:** No TBD/TODO; every code step shows the exact code; every run step shows the command and expected output. ✓

**Type consistency:** `voxelHalfExtent()` declared (`lod.hpp`), defined (`lod.cpp`), and called identically in `raytracing.cpp` (radius + emit log). `pt_lod_voxel` (options) → `lod_voxel` (raytracing) mapping is set once in `engine.cpp:237` and read in `engine.cpp:2537` + `raytracing.cpp:1621`. Names consistent throughout. ✓
