# `--lod N` Grid-Aggregation LOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible `--lod N` benchmark knob to the path tracer that aggregates particles into an N³ voxel grid, emitting one cell-center sphere per occupied cell, cutting BVH primitive count to attack BLAS build cost, trace cost, and transparency-noise together.

**Architecture:** The LOD lives entirely upstream of the BVH. When `pt_lod_cells > 0`, the per-frame AABB-generation stage is replaced by three GPU passes (clear → scatter → emit) that write a compacted list of occupied-cell spheres into the AABB buffer; the count is read back to the host and the BLAS/TLAS are full-rebuilt over exactly that many primitives. All of this is contained in `RayTracingContext::recordUpdateScene` via an internal one-shot submit — no engine frame-loop change, no trace-shader change. `pt_lod_cells == 0` (default) keeps today's exact per-particle path.

**Tech Stack:** C++17, Vulkan 1.3 (KHR ray tracing / acceleration structures, buffer-device-address), Slang compute shaders (runtime-compiled), CUDA interop for positions.

## Global Constraints

- Scene domain is the fixed **[-1,1]³** cube (all shaders hardcode `domainMin = -1`, `cellSize = 2/N`); particles outside are clamped into edge cells.
- `--lod N` = **cells per axis** (N³ grid). Reject `N > 512` at CLI parse (phase-1 memory cap: 512³·4 B = 537 MB accumulator).
- **Cell-center** placement only (no centroid): the scatter pass accumulates a `uint` occupancy count per cell and nothing else. Determinism comes from integer-count atomics + index-derived positions.
- LOD mode **always full-rebuilds** the BLAS (no refit). The BLAS is sized once for `maxCells = min(N³, P)`; each per-frame build uses `primitiveCount = read-back count ≤ maxCells`.
- BDA accesses (positions, AABBs) MUST use explicit 64-bit address arithmetic and a cast, never pointer indexing — the NVIDIA compiler truncates `OpPtrAccessChain` offsets to 32 bits past 4 GiB (see `shaders/pathtrace_aabbs.slang`).
- Atomic buffers (cell counts, global counter) are bound as `RWStructuredBuffer<uint>` descriptors (both ≤ 537 MB, under `maxStorageBufferRange`); only the huge positions/AABB buffers ride BDA.
- The feature is a **library capability** exposed via `ViewerOptions::pt_lod_cells`; `rr-server.cu` is an ordinary consumer.
- Build commands: library `./mimir-build-from-change.sh`; sample relink `./samples-build-from-change.sh --sample remote-rendering`. Clock-skew "modification time in the future" warnings are harmless here.

### Reference run commands

Fast dev smoke test (builds in seconds, exercises all logging + determinism):
```bash
samples/remote-rendering/build/rr-server 9000 1920 1080 $((2**20)) 413111 10000 \
  --pcolor 1.0,0.05,0.05 --background 0.2 --k 64 --epsilon 0.07 \
  --light-model path-tracing --spp 1 --size 0.1 --steps-per-frame 1 --fps 60 --fly --bounces 4 --lod 32
```
Full-scale check (the 2²⁹ workload from the investigation):
```bash
samples/remote-rendering/build/rr-server 9000 1920 1080 $((2**29)) 413111 10000 \
  --pcolor 1.0,0.05,0.05 --background 0.2 --k 64 --epsilon 0.07 \
  --light-model path-tracing --spp 1 --size 0.1 --steps-per-frame 1 --fps 60 --fly --bounces 4 --lod 128
```
The server logs stats without a client, but connect `rr-client` (with the raised `--first-frame-timeout`) to see the image. There is no GPU unit-test harness in this repo, so every task verifies by building and observing server log lines.

---

## File Structure

- `lib/include/public/mimir/options.hpp` — add `pt_lod_cells` to `ViewerOptions`.
- `samples/remote-rendering/rr-server.cu` — parse `--lod`, usage text, set option.
- `lib/include/private/mimir/raytracing.hpp` — new members (config, buffers, descriptor handles, pipelines, prim count) + method decls.
- `shaders/pathtrace_lod_scatter.slang` — NEW: bin particles → per-cell count.
- `shaders/pathtrace_lod_emit.slang` — NEW: occupied cell → compacted AABB.
- `lib/src/raytracing.cpp` — pipeline creation, buffer/descriptor allocation in `bindScene`, LOD branch in `recordUpdateScene`, `recordBlasBuildChunks` override param, teardown.
- `lib/src/engine.cpp` — copy `options.pt_lod_cells` into the RT context before `bindScene`.
- `samples/remote-rendering/README.md` — document `--lod`.

---

## Task 1: Option + CLI plumbing (no behavior change)

Wire `--lod N` from CLI to a stored `lod_cells` on the RT context and log it, without changing any geometry yet. Deliverable: `--lod` parses, validates, and logs; the rendered scene is byte-for-byte unchanged (LOD path not yet taken).

**Files:**
- Modify: `lib/include/public/mimir/options.hpp:165-171`
- Modify: `samples/remote-rendering/rr-server.cu` (arg parse ~260-275, usage ~142-166, option set ~380-395)
- Modify: `lib/include/private/mimir/raytracing.hpp` (config members block)
- Modify: `lib/src/engine.cpp:200-206` (before `bindScene`)
- Modify: `lib/src/raytracing.cpp:1140-1142` (bindScene logging)

**Interfaces:**
- Produces: `ViewerOptions::pt_lod_cells` (`unsigned int`, default 0); `RayTracingContext::lod_cells` (`uint32_t`, default 0).

- [ ] **Step 1: Add the option field**

In `lib/include/public/mimir/options.hpp`, after `pt_denoise` (line ~171):
```cpp
    // Level-of-detail for path tracing: N = cells per axis of an N^3 voxel grid over the [-1,1]^3
    // domain. 0 (default) = one primitive per particle (no LOD). N>0 aggregates particles into one
    // sphere per occupied cell, trading fidelity for BVH build+trace speed. Capped at 512 (--lod).
    unsigned int pt_lod_cells = 0;
```

- [ ] **Step 2: Add the RT context member**

In `lib/include/private/mimir/raytracing.hpp`, in the config members near `rebuild_interval` (~line 138):
```cpp
    // LOD grid resolution (cells per axis); 0 = per-particle (no LOD). Set from
    // ViewerOptions::pt_lod_cells before bindScene. See DESIGN_lod.md.
    uint32_t lod_cells = 0;
```

- [ ] **Step 3: Copy the option into the RT context before bindScene**

In `lib/src/engine.cpp`, immediately before the `raytracing.bindScene(...)` call (~line 205):
```cpp
                raytracing.lod_cells = engine.options.pt_lod_cells;
                raytracing.bindScene(pos_addr, view->draw_count, view->desc.default_size,
```
(Keep the existing `bindScene(...)` arguments unchanged; only add the assignment line directly above it. Verify the surrounding identifier is `engine.options` — match whatever the local variable is at that call site.)

- [ ] **Step 4: Parse `--lod` in the sample and validate**

In `samples/remote-rendering/rr-server.cu`, add a local near the other option locals (~line 227):
```cpp
    unsigned int lod_cells   = 0;
```
In the flag-parsing else-if chain (~line 260, alongside `--k` / `--epsilon`):
```cpp
            else if (a == "--lod")         lod_cells = (unsigned int)std::stoul(v);
```
After the parse loop, before building `ViewerOptions` (~line 340), validate:
```cpp
    if (lod_cells > 512) {
        fprintf(stderr, "rr-server: --lod must be 0..512 (got %u); phase-1 memory cap\n", lod_cells);
        return EXIT_FAILURE;
    }
```

- [ ] **Step 5: Set the option and document it**

Where `ViewerOptions` fields are assigned (near `.pt_max_bounces` / the options struct, ~line 385):
```cpp
    options.pt_lod_cells = lod_cells;
```
Add to the usage text (~line 166, after the `--bounces` line):
```cpp
        "  --lod N            Path-trace LOD: N^3 voxel grid, one sphere per occupied cell\n"
        "                     (0 = per-particle, default). 0..512. Trades detail for speed.\n"
```

- [ ] **Step 6: Log the request in bindScene**

In `lib/src/raytracing.cpp`, right after the "Path tracing: {} particles in {} BLAS chunk(s)" log (~line 1142):
```cpp
    if (lod_cells > 0)
    {
        spdlog::info("Path tracing: LOD grid requested: {}^3 cells (per-particle path still active until wired)", lod_cells);
    }
```
(This temporary "still active" wording is replaced in Task 3.)

- [ ] **Step 7: Build**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering`
Expected: both build; final line `Done. Binary/binaries in: .../remote-rendering/build/`.

- [ ] **Step 8: Run — valid and invalid**

Run the fast smoke command with `--lod 32` (see Reference run commands).
Expected: server starts; log contains `LOD grid requested: 32^3 cells`. The rendered scene is unchanged from `--lod 0`.
Run again with `--lod 999`.
Expected: exits immediately with `--lod must be 0..512 (got 999)`.

- [ ] **Step 9: Commit**

```bash
git add lib/include/public/mimir/options.hpp lib/include/private/mimir/raytracing.hpp \
        lib/src/engine.cpp lib/src/raytracing.cpp samples/remote-rendering/rr-server.cu
git commit -m "feat(lod): add --lod option, CLI parse/validate, and plumbing (no behavior yet)"
```

---

## Task 2: LOD compute shaders + pipelines

Add the scatter and emit Slang shaders and create their compute pipelines at init, unconditionally (cheap; they're only dispatched when LOD is active in Task 3). Deliverable: pipelines compile at runtime with no errors on every run.

**Files:**
- Create: `shaders/pathtrace_lod_scatter.slang`
- Create: `shaders/pathtrace_lod_emit.slang`
- Modify: `lib/include/private/mimir/raytracing.hpp` (push structs, pipeline/layout/descriptor handles, `createLodPipelines` decl)
- Modify: `lib/src/raytracing.cpp` (implement `createLodPipelines`, call it from init, destroy in teardown)

**Interfaces:**
- Consumes: nothing from earlier tasks beyond `lod_cells`.
- Produces:
  - Shader entry points `scatterMain` (`shaders/pathtrace_lod_scatter.slang`) and `emitMain` (`shaders/pathtrace_lod_emit.slang`).
  - `RayTracingContext::lod_scatter_pipeline`, `lod_scatter_layout`, `lod_scatter_set_layout`; `lod_emit_pipeline`, `lod_emit_layout`, `lod_emit_set_layout`; `lod_desc_pool`; `lod_scatter_set`, `lod_emit_set` (all `VkDescriptorSet`, written in Task 3).
  - `struct LodScatterPush { VkDeviceAddress positions; uint32_t count; uint32_t gridN; };`
  - `struct LodEmitPush { VkDeviceAddress aabbs; uint32_t gridN; float radius; };`
  - `void RayTracingContext::createLodPipelines();`

- [ ] **Step 1: Write the scatter shader**

Create `shaders/pathtrace_lod_scatter.slang`:
```slang
// LOD grid-aggregation, scatter pass. Bins each particle into an N^3 grid over the fixed
// [-1,1]^3 domain and atomically increments the per-cell occupancy count. Count only: the
// representative sphere is placed at the cell centre by the emit pass, so no position sum is
// needed. Integer-count atomics are order-independent => deterministic occupied-cell set.

// Positions ride the push constants as a raw buffer-device-address pointer (the buffer is many
// GiB at large N; a descriptor is capped by maxStorageBufferRange). The per-cell count buffer is
// small (<= 537 MB at 512^3) so it is a bound RWStructuredBuffer that supports InterlockedAdd.
struct PushConstants
{
    float* positions; // in: interop positions, tightly packed x,y,z (stride 12 B). BDA, offset 0.
    uint   count;     // number of particles
    uint   gridN;     // cells per axis
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> cellCounts; // out: one uint per cell (N^3)

[shader("compute")]
[numthreads(64, 1, 1)]
void scatterMain(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.count) { return; }

    // Explicit 64-bit BDA address arithmetic + cast (NVIDIA truncates OpPtrAccessChain offsets to
    // 32 bits past 4 GiB; see pathtrace_aabbs.slang).
    uint64_t pos_addr = uint64_t(pc.positions) + uint64_t(i) * 12ull;
    float px = *(float*)(pos_addr + 0);
    float py = *(float*)(pos_addr + 4);
    float pz = *(float*)(pos_addr + 8);

    float n = float(pc.gridN);
    // Fixed domain [-1,1] -> [0,1] -> cell index, clamped to [0, N-1].
    int cx = int(clamp((px + 1.0) * 0.5 * n, 0.0, n - 1.0));
    int cy = int(clamp((py + 1.0) * 0.5 * n, 0.0, n - 1.0));
    int cz = int(clamp((pz + 1.0) * 0.5 * n, 0.0, n - 1.0));
    uint lin = uint(cx) + pc.gridN * (uint(cy) + pc.gridN * uint(cz));

    uint old;
    InterlockedAdd(cellCounts[lin], 1u, old);
}
```

- [ ] **Step 2: Write the emit shader**

Create `shaders/pathtrace_lod_emit.slang`:
```slang
// LOD grid-aggregation, emit/compact pass. One thread per cell (N^3 total). Occupied cells append
// one sphere AABB (at the cell's geometric centre, radius sized to the cell) into a compacted list
// via a global atomic counter. The slot order is nondeterministic but the count and the resulting
// BVH/image are not (order-independent), so benchmark metrics stay reproducible.

// Matches VkAabbPositionsKHR: 6 tightly packed floats (24 bytes), min xyz then max xyz.
struct Aabb { float minx, miny, minz, maxx, maxy, maxz; };

struct PushConstants
{
    Aabb* aabbs;  // out: compacted AABB list. BDA, offset 0.
    uint  gridN;  // cells per axis
    float radius; // sphere world radius (= coverage * cellSize / 2)
};
[[vk::push_constant]] PushConstants pc;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> cellCounts;  // in : per-cell occupancy
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> globalCount; // in/out: element 0 = # emitted

[shader("compute")]
[numthreads(64, 1, 1)]
void emitMain(uint3 tid : SV_DispatchThreadID)
{
    uint lin   = tid.x;
    uint total = pc.gridN * pc.gridN * pc.gridN; // 512^3 = 134,217,728 fits in uint32
    if (lin >= total) { return; }
    if (cellCounts[lin] == 0u) { return; }

    uint gx = lin % pc.gridN;
    uint gy = (lin / pc.gridN) % pc.gridN;
    uint gz = lin / (pc.gridN * pc.gridN);
    float cs = 2.0 / float(pc.gridN);
    // Cell geometric centre in [-1,1]^3.
    float3 c = float3(-1.0, -1.0, -1.0) + (float3(float(gx), float(gy), float(gz)) + 0.5) * cs;

    uint slot;
    InterlockedAdd(globalCount[0], 1u, slot);

    Aabb a;
    a.minx = c.x - pc.radius; a.miny = c.y - pc.radius; a.minz = c.z - pc.radius;
    a.maxx = c.x + pc.radius; a.maxy = c.y + pc.radius; a.maxz = c.z + pc.radius;
    // Explicit 64-bit BDA address + cast (see pathtrace_aabbs.slang).
    *(Aabb*)(uint64_t(pc.aabbs) + uint64_t(slot) * 24ull) = a;
}
```

- [ ] **Step 3: Declare push structs, handles, and the create method**

In `lib/include/private/mimir/raytracing.hpp`, near the other pipeline handles (e.g. `iw_pipeline` / `atrous_pipeline`), add:
```cpp
    // ---- LOD grid-aggregation compute pipelines (pt_lod_cells > 0) ----
    VkDescriptorSetLayout lod_scatter_set_layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout lod_emit_set_layout    = VK_NULL_HANDLE;
    VkPipelineLayout      lod_scatter_layout     = VK_NULL_HANDLE;
    VkPipelineLayout      lod_emit_layout        = VK_NULL_HANDLE;
    VkPipeline            lod_scatter_pipeline   = VK_NULL_HANDLE;
    VkPipeline            lod_emit_pipeline       = VK_NULL_HANDLE;
    VkDescriptorPool      lod_desc_pool          = VK_NULL_HANDLE;
    VkDescriptorSet       lod_scatter_set        = VK_NULL_HANDLE; // written in bindScene
    VkDescriptorSet       lod_emit_set           = VK_NULL_HANDLE; // written in bindScene

    void createLodPipelines();
```

- [ ] **Step 4: Implement `createLodPipelines`**

In `lib/src/raytracing.cpp`, after `createAtrousPipeline` (~line 802), add the push structs and the function. The pipeline-creation boilerplate mirrors `createAtrousPipeline`/`createAabbWriter`:
```cpp
// ---- LOD grid-aggregation compute pipelines ---------------------------------------

// Push constants for pathtrace_lod_scatter.slang: positions BDA pointer (offset 0), then the
// particle count and grid resolution. The per-cell count buffer is a descriptor (binding 0).
struct LodScatterPush { VkDeviceAddress positions; uint32_t count; uint32_t gridN; };
// Push constants for pathtrace_lod_emit.slang: AABB output BDA pointer (offset 0), grid
// resolution, sphere radius. Cell counts (binding 0) and the global counter (binding 1) are
// descriptors.
struct LodEmitPush { VkDeviceAddress aabbs; uint32_t gridN; float radius; };

void RayTracingContext::createLodPipelines()
{
    // One-binding set for scatter (cell counts), two-binding set for emit (cell counts + counter).
    auto make_set_layout = [&](uint32_t binding_count) {
        std::vector<VkDescriptorSetLayoutBinding> b(binding_count);
        for (uint32_t i = 0; i < binding_count; ++i) {
            b[i] = VkDescriptorSetLayoutBinding{
                .binding = i, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                .pImmutableSamplers = nullptr };
        }
        VkDescriptorSetLayoutCreateInfo info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr, .flags = 0, .bindingCount = binding_count, .pBindings = b.data() };
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        validation::checkVulkan(vkCreateDescriptorSetLayout(device, &info, nullptr, &layout));
        return layout;
    };
    lod_scatter_set_layout = make_set_layout(1);
    lod_emit_set_layout    = make_set_layout(2);

    auto make_pipeline = [&](VkDescriptorSetLayout set_layout, uint32_t push_size,
                             const char* module, const char* entry,
                             VkPipelineLayout& out_layout, VkPipeline& out_pipe) {
        VkPushConstantRange range{ .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = push_size };
        VkPipelineLayoutCreateInfo li{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pNext = nullptr, .flags = 0,
            .setLayoutCount = 1, .pSetLayouts = &set_layout,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &range };
        validation::checkVulkan(vkCreatePipelineLayout(device, &li, nullptr, &out_layout));

        auto orig = std::filesystem::current_path();
        std::filesystem::current_path(getDefaultShaderPath());
        auto builder = ShaderBuilder::make();
        ShaderCompileParams params{ .module_path = module, .entrypoints = { entry }, .specializations = {} };
        auto stages = builder.compileModule(device, params);
        std::filesystem::current_path(orig);
        if (stages.size() != 1) { spdlog::error("{}: expected 1 compute stage, got {}", module, stages.size()); }

        VkComputePipelineCreateInfo pi{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .pNext = nullptr, .flags = 0,
            .stage = stages[0], .layout = out_layout,
            .basePipelineHandle = VK_NULL_HANDLE, .basePipelineIndex = 0 };
        validation::checkVulkan(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pi, nullptr, &out_pipe));
        vkDestroyShaderModule(device, stages[0].module, nullptr);
    };
    make_pipeline(lod_scatter_set_layout, sizeof(LodScatterPush),
        "shaders/pathtrace_lod_scatter.slang", "scatterMain", lod_scatter_layout, lod_scatter_pipeline);
    make_pipeline(lod_emit_set_layout, sizeof(LodEmitPush),
        "shaders/pathtrace_lod_emit.slang", "emitMain", lod_emit_layout, lod_emit_pipeline);

    // One set of each (the aggregate runs in an internal one-shot submit, not per-frame-in-flight).
    VkDescriptorPoolSize pool_size{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 }; // 1 (scatter) + 2 (emit)
    VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .pNext = nullptr, .flags = 0,
        .maxSets = 2, .poolSizeCount = 1, .pPoolSizes = &pool_size };
    validation::checkVulkan(vkCreateDescriptorPool(device, &pool_info, nullptr, &lod_desc_pool));

    VkDescriptorSetLayout layouts[2] = { lod_scatter_set_layout, lod_emit_set_layout };
    VkDescriptorSet sets[2] = {};
    VkDescriptorSetAllocateInfo ai{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr,
        .descriptorPool = lod_desc_pool, .descriptorSetCount = 2, .pSetLayouts = layouts };
    validation::checkVulkan(vkAllocateDescriptorSets(device, &ai, sets));
    lod_scatter_set = sets[0];
    lod_emit_set    = sets[1];
}
```

- [ ] **Step 5: Call it at init and tear it down**

Find where `createAabbWriter(*this)` / `createAtrousPipeline(*this)` are called during path-tracing init and add a call after them:
```cpp
    createLodPipelines();
```
(These are member-style free functions in the existing code; `createLodPipelines` is a method, so call it as `createLodPipelines();` from inside the RT context init method, or `ctx.createLodPipelines();` if called from a free function with `ctx`.)

In the RT teardown (`destroy()` / the function that destroys `iw_pipeline`, `atrous_pipeline`, etc.), add:
```cpp
    if (lod_scatter_pipeline   != VK_NULL_HANDLE) { vkDestroyPipeline(device, lod_scatter_pipeline, nullptr); }
    if (lod_emit_pipeline      != VK_NULL_HANDLE) { vkDestroyPipeline(device, lod_emit_pipeline, nullptr); }
    if (lod_scatter_layout     != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, lod_scatter_layout, nullptr); }
    if (lod_emit_layout        != VK_NULL_HANDLE) { vkDestroyPipelineLayout(device, lod_emit_layout, nullptr); }
    if (lod_scatter_set_layout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, lod_scatter_set_layout, nullptr); }
    if (lod_emit_set_layout    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(device, lod_emit_set_layout, nullptr); }
    if (lod_desc_pool          != VK_NULL_HANDLE) { vkDestroyDescriptorPool(device, lod_desc_pool, nullptr); }
```

- [ ] **Step 6: Build**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering`
Expected: both build cleanly.

- [ ] **Step 7: Run — verify shaders compile at runtime**

Run the fast smoke command (with or without `--lod`; pipelines are created unconditionally).
Expected: NO `pathtrace_lod_scatter.slang: expected 1 compute stage` or `pathtrace_lod_emit.slang: ...` errors and NO Vulkan validation errors in the log; server reaches "listening on port 9000". (The pipelines exist but are not dispatched yet.)

- [ ] **Step 8: Commit**

```bash
git add shaders/pathtrace_lod_scatter.slang shaders/pathtrace_lod_emit.slang \
        lib/include/private/mimir/raytracing.hpp lib/src/raytracing.cpp
git commit -m "feat(lod): add scatter/emit compute shaders and pipelines"
```

---

## Task 3: LOD buffers + aggregation + build switch (functional)

Allocate the LOD buffers, size the AABB buffer and BLAS for `maxCells`, write the descriptor sets, and replace the AABB-generation stage with the three-pass aggregation + count read-back + full rebuild when `lod_cells > 0`. Deliverable: `--lod N` renders a recognizable aggregated cloud, logs the occupied-cell count, is deterministic, and drops build/trace time; `--lod 0` is unchanged.

**Files:**
- Modify: `lib/include/private/mimir/raytracing.hpp` (buffer/count members + `recordLodUpdate` decl + constant)
- Modify: `lib/src/raytracing.cpp:163-256` (`recordBlasBuildChunks` override param), `:1114-1142` (bindScene sizing), `:1160-1200` (descriptor writes), `:1212-1344` (recordUpdateScene branch + `recordLodUpdate`), teardown
- Test: drive-the-app (server log + rr-client image)

**Interfaces:**
- Consumes: `LodScatterPush`, `LodEmitPush`, `lod_scatter_pipeline`/`lod_emit_pipeline`/`lod_scatter_layout`/`lod_emit_layout`/`lod_scatter_set`/`lod_emit_set` (Task 2); `lod_cells` (Task 1); existing `submit`, `makeBuffer`, `aabb_buffer`, `position_address`, `particle_count`, `recordBlasBuildChunks`, `recordTlasBuild`, `blas_chunk_prims`, `scene_blas`, `scene_tlas`, `tlas_scratch`, `tlas_instance_buffer`.
- Produces:
  - `RtBuffer lod_cellcount_buffer;` (N³·4 B, DEVICE_LOCAL storage)
  - `RtBuffer lod_counter_buffer;` (4 B, HOST_VISIBLE storage, address)
  - `uint32_t lod_max_cells = 0;` (= min(N³, P))
  - `uint32_t lod_prim_count = 0;` (occupied cells this frame)
  - `static constexpr float LOD_COVERAGE = 1.2f;`
  - `void RayTracingContext::recordLodUpdate(VkCommandBuffer cmd, uint32_t frame_idx);`
  - `recordBlasBuildChunks(ctx, cmd, aabb_addr, update, override_prims=0)` — new trailing param.

- [ ] **Step 1: Add buffer/count members and the constant**

In `lib/include/private/mimir/raytracing.hpp`, near the LOD pipeline handles from Task 2:
```cpp
    RtBuffer lod_cellcount_buffer; // N^3 uint occupancy counts (DEVICE_LOCAL)
    RtBuffer lod_counter_buffer;   // 1 uint emitted-primitive counter (HOST_VISIBLE, readback)
    uint32_t lod_max_cells  = 0;   // min(N^3, particle_count): BLAS/AABB sizing bound
    uint32_t lod_prim_count = 0;   // occupied cells emitted this frame (per-frame build count)
    static constexpr float LOD_COVERAGE = 1.2f; // sphere radius = LOD_COVERAGE * cellSize / 2
    void recordLodUpdate(VkCommandBuffer cmd, uint32_t frame_idx);
```

- [ ] **Step 2: Add an override-primitive-count param to `recordBlasBuildChunks`**

In `lib/src/raytracing.cpp`, change the signature (~line 216) and the per-chunk prim count (~line 225). Update the forward declaration if one exists near the top of the file.
```cpp
void recordBlasBuildChunks(RayTracingContext& ctx, VkCommandBuffer cmd, VkDeviceAddress aabb_addr,
    bool update, uint32_t override_prims = 0)
{
    auto scratch_align = ctx.accel_props.minAccelerationStructureScratchOffsetAlignment;
    VkDeviceAddress scratch_addr = alignUp(ctx.blas_scratch.address, scratch_align);
    uint32_t num_chunks = static_cast<uint32_t>(ctx.scene_blas.size());

    for (uint32_t c = 0; c < num_chunks; ++c)
    {
        // LOD builds a single chunk over the compacted occupied-cell list, whose length varies per
        // frame; override_prims (> 0) supplies that count. The per-particle path passes 0 and uses
        // the fixed chunk size.
        uint32_t prims = (override_prims > 0) ? override_prims : chunkPrimCount(ctx, c);
```
(The rest of the loop body is unchanged — `prims` continues to feed `VkAccelerationStructureBuildRangeInfoKHR::primitiveCount`.)

- [ ] **Step 3: Size AABB buffer + BLAS for maxCells and allocate LOD buffers in bindScene**

In `lib/src/raytracing.cpp`, replace the AABB buffer sizing block (~lines 1114-1129) so LOD uses `maxCells`:
```cpp
    // LOD aggregates particles into <= min(N^3, P) occupied-cell spheres, so both the AABB buffer
    // and the BLAS are sized to that bound (smaller than P at large N). Per-particle mode sizes to P.
    uint32_t geom_prims = count;
    if (lod_cells > 0)
    {
        uint64_t cells = uint64_t(lod_cells) * lod_cells * lod_cells;
        lod_max_cells = static_cast<uint32_t>(std::min<uint64_t>(cells, count));
        geom_prims = lod_max_cells;

        // Per-cell occupancy counts (one uint each) + the emitted-primitive counter (host-readable).
        lod_cellcount_buffer = makeBuffer(*this, VkDeviceSize(cells) * sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DEVICE_LOCAL, false);
        lod_counter_buffer = makeBuffer(*this, sizeof(uint32_t),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_VISIBLE, true);
    }

    VkDeviceSize aabb_size = VkDeviceSize(geom_prims) * sizeof(VkAabbPositionsKHR);
    aabb_buffer = makeBuffer(*this, aabb_size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        DEVICE_LOCAL, true);

    blas_chunk_prims = accel_props.maxPrimitiveCount > 0
        ? static_cast<uint32_t>(std::min<uint64_t>(accel_props.maxPrimitiveCount, geom_prims))
        : geom_prims;
```
Then change the BLAS/log lines just below to use `geom_prims`:
```cpp
    createDynamicBlasChunks(*this, aabb_buffer.address, geom_prims);
    uint32_t num_chunks = static_cast<uint32_t>(scene_blas.size());
    spdlog::info("Path tracing: {} particles in {} BLAS chunk(s) of up to {} prims",
        count, num_chunks, blas_chunk_prims);
    if (lod_cells > 0)
    {
        spdlog::info("Path tracing: LOD {}^3 grid, up to {} occupied-cell primitives (from {} particles)",
            lod_cells, lod_max_cells, count);
    }
```
Remove the temporary "still active" log from Task 1 Step 6.

Because `geom_prims = min(N^3, P) <= 134 M < 2^29`, LOD is always a single BLAS chunk (`num_chunks == 1`), so the existing chunk-loop and instance-loop below stay correct with one chunk.

- [ ] **Step 4: Write the LOD descriptor sets in bindScene**

In `lib/src/raytracing.cpp`, after the per-frame TLAS descriptor writes (~line 1200), before the initial `submit(... recordUpdateScene ...)`:
```cpp
    if (lod_cells > 0)
    {
        VkDescriptorBufferInfo cc{ .buffer = lod_cellcount_buffer.buffer, .offset = 0, .range = VK_WHOLE_SIZE };
        VkDescriptorBufferInfo gc{ .buffer = lod_counter_buffer.buffer,  .offset = 0, .range = VK_WHOLE_SIZE };
        VkWriteDescriptorSet writes[3] = {
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = lod_scatter_set,
              .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pImageInfo = nullptr,
              .pBufferInfo = &cc, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = lod_emit_set,
              .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pImageInfo = nullptr,
              .pBufferInfo = &cc, .pTexelBufferView = nullptr },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .pNext = nullptr, .dstSet = lod_emit_set,
              .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pImageInfo = nullptr,
              .pBufferInfo = &gc, .pTexelBufferView = nullptr },
        };
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
    }
```

- [ ] **Step 5: Branch `recordUpdateScene` to the LOD path**

In `lib/src/raytracing.cpp`, at the top of `recordUpdateScene` (after the `must_build` computation, ~line 1219), before the existing skip/serialize logic:
```cpp
    if (lod_cells > 0)
    {
        if (!must_build)
        {
            // Same skip bookkeeping as the per-particle path: reuse the existing AS, stamp ~0 build.
            stat_skips++;
            slot_build_mode[frame_idx] = BlasBuild::Skip;
            if (timing_pool != VK_NULL_HANDLE)
            {
                vkCmdResetQueryPool(cmd, timing_pool, frame_idx * TS_PER_FRAME, TS_PER_FRAME);
                uint32_t base = frame_idx * TS_PER_FRAME;
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,    timing_pool, base + 0);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 1);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 2);
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, base + 3);
            }
            return;
        }
        recordLodUpdate(cmd, frame_idx);
        return;
    }
```

- [ ] **Step 6: Implement `recordLodUpdate`**

In `lib/src/raytracing.cpp`, add after `recordUpdateScene` (~line 1344):
```cpp
// LOD path: aggregate particles into occupied-cell spheres on the GPU (clear -> scatter -> emit)
// in an internal one-shot submit, read the emitted count back, then full-rebuild the BLAS/TLAS over
// exactly that many primitives in the frame command buffer. The one-shot's vkQueueWaitIdle also
// serializes against the previous frame's trace (which reads the AABB buffer), so no separate
// cross-frame barrier is needed here. Aggregate time is not itemized in the GPU sub-phase split
// (it runs outside `cmd`); it shows up in the total wall-clock render time.
void RayTracingContext::recordLodUpdate(VkCommandBuffer cmd, uint32_t frame_idx)
{
    const uint32_t grid = lod_cells;
    const uint64_t num_cells = uint64_t(grid) * grid * grid;
    const float cell_size = 2.0f / float(grid);
    const float radius = LOD_COVERAGE * cell_size * 0.5f;

    // 1) Aggregate in a blocking one-shot submit.
    submit([&](VkCommandBuffer c) {
        vkCmdFillBuffer(c, lod_cellcount_buffer.buffer, 0, VK_WHOLE_SIZE, 0u);
        vkCmdFillBuffer(c, lod_counter_buffer.buffer,   0, VK_WHOLE_SIZE, 0u);
        VkMemoryBarrier clr{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &clr, 0, nullptr, 0, nullptr);

        LodScatterPush sp{ .positions = position_address, .count = particle_count, .gridN = grid };
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_scatter_pipeline);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_scatter_layout, 0, 1,
            &lod_scatter_set, 0, nullptr);
        vkCmdPushConstants(c, lod_scatter_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
        vkCmdDispatch(c, (particle_count + 63) / 64, 1, 1);

        VkMemoryBarrier s2e{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &s2e, 0, nullptr, 0, nullptr);

        LodEmitPush ep{ .aabbs = aabb_buffer.address, .gridN = grid, .radius = radius };
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_emit_pipeline);
        vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, lod_emit_layout, 0, 1,
            &lod_emit_set, 0, nullptr);
        vkCmdPushConstants(c, lod_emit_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ep), &ep);
        vkCmdDispatch(c, static_cast<uint32_t>((num_cells + 63) / 64), 1, 1);
    });

    // 2) Read the emitted primitive count (HOST_VISIBLE, coherent).
    uint32_t occupied = 0;
    void* mapped = nullptr;
    validation::checkVulkan(vkMapMemory(device, lod_counter_buffer.memory, 0, sizeof(uint32_t), 0, &mapped));
    std::memcpy(&occupied, mapped, sizeof(uint32_t));
    vkUnmapMemory(device, lod_counter_buffer.memory);
    occupied = std::min(occupied, lod_max_cells);
    lod_prim_count = occupied;

    // 3) Build the AS in the frame command buffer over `occupied` primitives (always a full rebuild).
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdResetQueryPool(cmd, timing_pool, frame_idx * TS_PER_FRAME, TS_PER_FRAME);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 0);
        // Aggregate ran outside `cmd`; mark the AABB sub-phase as ~0 here.
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 1);
    }

    recordBlasBuildChunks(*this, cmd, aabb_buffer.address, /*update=*/false, /*override_prims=*/occupied);
    accel_ever_built = true;
    frames_since_full_rebuild = 0;
    stat_full_rebuilds++;
    slot_build_mode[frame_idx] = BlasBuild::Rebuild;

    VkMemoryBarrier blas_to_tlas{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &blas_to_tlas, 0, nullptr, 0, nullptr);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 2);
    }

    recordTlasBuild(*this, cmd, scene_tlas, tlas_scratch, tlas_instance_buffer.address,
        static_cast<uint32_t>(scene_blas.size()));

    VkMemoryBarrier to_trace{ .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
        .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, 1, &to_trace, 0, nullptr, 0, nullptr);
    if (timing_pool != VK_NULL_HANDLE)
    {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, timing_pool, frame_idx * TS_PER_FRAME + 3);
    }

    // Log the aggregation result once per bind (frame_idx 0 is the initial bindScene build).
    if (frame_idx == 0)
    {
        spdlog::info("Path tracing: LOD emitted {} occupied cells (reduction {:.0f}:1 vs {} particles)",
            occupied, occupied ? double(particle_count) / double(occupied) : 0.0, particle_count);
    }
}
```

- [ ] **Step 7: Free the LOD buffers in teardown**

In the RT teardown, alongside the `aabb_buffer` free (search for where `aabb_buffer.buffer` is destroyed), add frees for `lod_cellcount_buffer` and `lod_counter_buffer` using the same `destroyBuffer`/`vkDestroyBuffer`+`vkFreeMemory` pattern the file already uses for `RtBuffer`s. (Match the existing helper — if the file frees `RtBuffer`s via a helper like `destroyBuffer(ctx, buf)`, use it; otherwise mirror the explicit `vkDestroyBuffer` + `vkFreeMemory` calls.)

- [ ] **Step 8: Build**

Run: `./mimir-build-from-change.sh && ./samples-build-from-change.sh --sample remote-rendering`
Expected: both build cleanly.

- [ ] **Step 9: Run — small-N functional + determinism**

Run the fast smoke command with `--lod 32` (2²⁰ particles).
Expected in the log:
- `Path tracing: LOD 32^3 grid, up to N occupied-cell primitives ...`
- `Path tracing: LOD emitted M occupied cells (reduction R:1 vs 1048576 particles)` with `0 < M <= 32768` and `M <= 1048576`.
- No Vulkan validation errors.
Run it a **second** time and confirm the emitted count `M` is **identical** (determinism).
Run once more with `--lod 0`.
Expected: no LOD log lines; behaves exactly as before this task.

- [ ] **Step 10: Run — full-scale image + speed**

Start the full-scale command with `--lod 128` (2²⁹), connect `rr-client` (raised `--first-frame-timeout`).
Expected: the cloud renders as recognizable opaque blobs (not holes); the stats line's `blas` and `trace` ms are far below the per-particle numbers (7191 ms / 80541 ms at `--lod 0`); `LOD emitted` count is logged. Sweep `--lod 64 / 128 / 256` and confirm the emitted count and the times rise monotonically with N.

- [ ] **Step 11: Commit**

```bash
git add lib/include/private/mimir/raytracing.hpp lib/src/raytracing.cpp
git commit -m "feat(lod): grid aggregation, count readback, and full-rebuild switch"
```

---

## Task 4: Documentation

Document `--lod` for users. Deliverable: the sample README describes the knob, its semantics, cap, and the quality/speed trade.

**Files:**
- Modify: `samples/remote-rendering/README.md`

- [ ] **Step 1: Add a `--lod` section to the README**

Add under the path-tracing options:
```markdown
### `--lod N` (path-tracing level of detail)

Aggregates particles into an `N x N x N` voxel grid over the `[-1,1]^3` domain and renders
one sphere per **occupied** cell (placed at the cell centre, sized to the cell), instead of one
sphere per particle. This cuts the number of BVH primitives, reducing both the BLAS build time and
the trace time, and — because the cell spheres are opaque and overlapping — removes the
transparency noise that tiny per-particle spheres produce at high counts.

- `N = 0` (default): one primitive per particle (no LOD).
- `N` in `1..512`: `N` cells per axis. Larger `N` = finer detail, more primitives, slower.
- Deterministic: the same `N` yields the same occupied-cell count and image every run, so it is a
  reproducible benchmark knob.
- Memory: the grid accumulator is `N^3 * 4 bytes` (512^3 = 537 MB).

Example:
    rr-server 9000 1920 1080 $((2**29)) ... --light-model path-tracing --lod 128
```

- [ ] **Step 2: Commit**

```bash
git add samples/remote-rendering/README.md
git commit -m "docs(lod): document --lod in the remote-rendering README"
```

---

## Self-Review Notes

- **Spec coverage:** domain/[-1,1]³ (Task 2/3 shaders), N=cells-per-axis + ≤512 cap (Task 1), cell-center count-only (Task 2 shaders), 4 B/cell accumulator (Task 3 Step 3), compacted AABB = min(N³,P) (Task 3 Step 3), contained readback via internal one-shot submit (Task 3 Step 6), always-full-rebuild (Task 3 Step 6), determinism (Task 3 Step 9), `--lod 0` unchanged (Task 1 Step 8 / Task 3 Step 9), library-boundary via `ViewerOptions` (Task 1). Non-goal noted: LOD aggregate time is not itemized in the GPU sub-phase split (Task 3 Step 6 comment) — acceptable per spec's phase-1 scope.
- **Placeholder scan:** every code step contains the actual code; verification steps give exact commands + expected log lines. No TBD/TODO.
- **Type consistency:** `LodScatterPush`/`LodEmitPush`, `lod_scatter_pipeline`/`lod_emit_pipeline`, `lod_scatter_layout`/`lod_emit_layout`, `lod_scatter_set`/`lod_emit_set`, `lod_cellcount_buffer`/`lod_counter_buffer`, `lod_max_cells`/`lod_prim_count`, `recordLodUpdate`, and the `recordBlasBuildChunks(..., override_prims)` signature are named identically across Tasks 2–3.
- **Known integration risks to watch during execution:** (a) the exact init function that calls `createAabbWriter`/`createAtrousPipeline` and the exact teardown function — grep for those names and co-locate the new calls; (b) whether `RtBuffer`s are freed via a helper or explicit calls — match the file's convention; (c) `maxComputeWorkGroupCount[0]` must exceed `N^3/64` (2.1 M at 512³) — the existing per-particle writer already dispatches 8.4 M groups at 2²⁹ on this Blackwell card, so the limit is high enough, but confirm no validation error on the emit dispatch.
