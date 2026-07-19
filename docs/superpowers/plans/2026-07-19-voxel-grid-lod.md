# Voxels Grid-Coarsening LOD Implementation Plan (in-shader pooling)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--lod M` transparently coarsen a `ViewType::Voxels` grid — draw M³ cubes (M < N) whose per-cube state is **max-pooled from the fine N³ grid on the fly in the vertex shader**, materializing NO coarse data.

**Architecture:** No coarse buffers, no compute kernel, no extra interop sync. A specialized voxel pipeline draws **M³ procedural points** (`vkCmdDraw(M³,1,0,0)`, no vertex buffer): the vertex shader derives each coarse cell `(cx,cy,cz)` from `SV_VertexID`, computes its world position (same mapping as the fine structured grid), reads its disjoint `(N/M)³` block of the **fine** state buffer (bound as a storage buffer), takes the **max**, and looks up `colormap[max]`. The existing geometry shader expands the point to a cube unchanged. Because the shader reads the *fine* state directly, it reuses the exact sim→draw synchronization mimir already applies to the normal voxel color attribute — so the ping/pong pair of views "just works" (each view binds its own fine state; visibility stays with the caller).

**Tech Stack:** Slang (new `voxel_lod.slang`), Vulkan (graphics pipeline variant + storage-buffer descriptors + push constants), CUDA-interop (fine state buffer, read-only in the shader), C++20, spdlog, CMake.

## Global Constraints

- Driven entirely by the existing `--lod`/`options.pt_lod_cells` option + `ViewType::Voxels`. No new public API beyond documenting Voxels semantics.
- Pooling operator is **max** over each coarse cell's disjoint `(N/M)³` fine block (state 0 = dead; max = "most-alive present"). Computed in the vertex shader; nothing is written back.
- State element type is **`int32` (4 bytes)** (the colormap-index type, `index_size == sizeof(int)`); ignore LOD (draw fine) if `index_size != 4`.
- Coarse cell → fine range per axis is `[c*N/M, (c+1)*N/M)` with integer floor (handles non-divisible N/M). Mirror this EXACTLY in the shader and in any reference used for tests.
- `M` must satisfy `0 < M < N`; `M >= N` or `M == 0` ⇒ no-op (draw fine unchanged), never upsample.
- The world position of coarse cell `(cx,cy,cz)` MUST use the same per-axis mapping `makeStructuredGrid` uses for a K³ grid with K=M, so coarse cubes fill the same world box as the fine grid. Read that formula from `MimirInstance::makeStructuredGrid` and pass its origin/spacing to the shader via push constants — do not hardcode a guess.
- Reuse the existing `voxel.slang` geometry + fragment stages; only the VERTEX stage is new. Follow the pipeline-creation pattern in `lib/src/pipeline.cpp` (the `ViewType::Voxels` case) and the descriptor/push-constant pattern in `lib/src/lod.cpp` (`make_pipeline`).
- CUDA-written / shader-read ordering for the fine state is ALREADY handled by mimir's interop for the normal voxel draw — do NOT add new synchronization.

---

### Task 1: `voxel_lod.slang` — procedural M³ vertex stage that max-pools the fine grid

**Files:**
- Create: `shaders/voxel_lod.slang`
- Create: `lib/tests/voxel_pool_ref_test.cu` (a CPU vs GPU parity test for the pooling *formula* the shader implements — the oracle that guards correctness, since the shader itself is not unit-testable)
- Create: `lib/src/voxel_lod.cu` (a reference `voxelPoolMax` used only by the test + as an optional debug/verify path; NOT on the render hot path)
- Create: `lib/include/private/mimir/voxel_lod.hpp`
- Modify: `lib/CMakeLists.txt` (add `voxel_pool_ref_test` target mirroring `lod_reduce_test`)

**Interfaces:**
- Produces (for tests/verify only): `namespace mimir { void voxelPoolMax(const int* fine, uint32_t N, int* coarse, uint32_t M, cudaStream_t stream); }` — same max-pool the shader does, for validating the formula and for an optional `CA_LOD_CHECK`.
- Produces (shader): `voxel_lod.slang` exporting a `vertexLodMain` entry that consumes only `SV_VertexID` + push constants `{ uint fineN; uint coarseM; float4 gridOrigin; float4 gridSpacing; }` and two storage buffers `fineState : int[]`, `colormap : float4[]`, emitting the SAME `VertexData { float4 center; float4 color; }` the existing `voxel.slang` vertex stage emits (so the existing geometry stage links unchanged).

- [ ] **Step 1: Write the failing parity test for the pooling formula**

Create `lib/tests/voxel_pool_ref_test.cu` (identical structure to the reference the shader must match):
```cpp
#include "mimir/voxel_lod.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>

static int refCell(const std::vector<int>& f, uint32_t N, uint32_t M, uint32_t cx, uint32_t cy, uint32_t cz){
    auto lo=[&](uint32_t c){return (uint32_t)((uint64_t)c*N/M);}; auto hi=[&](uint32_t c){return (uint32_t)(((uint64_t)c+1)*N/M);};
    int m=0; for(uint32_t z=lo(cz);z<hi(cz);++z)for(uint32_t y=lo(cy);y<hi(cy);++y)for(uint32_t x=lo(cx);x<hi(cx);++x)
        m=std::max(m,f[(size_t)x+N*((size_t)y+(size_t)N*z)]); return m;
}
int main(){
    struct C{uint32_t N,M;} cs[]={{8,4},{9,4},{16,4},{100,25},{128,32},{130,31}};
    std::mt19937 rng(7);
    for(auto c:cs){
        uint64_t nc=(uint64_t)c.N*c.N*c.N, mc=(uint64_t)c.M*c.M*c.M;
        std::vector<int> f(nc); std::uniform_int_distribution<int> d(0,3); for(auto&v:f)v=d(rng);
        int *df,*dc; cudaMalloc(&df,nc*4); cudaMalloc(&dc,mc*4); cudaMemcpy(df,f.data(),nc*4,cudaMemcpyHostToDevice);
        mimir::voxelPoolMax(df,c.N,dc,c.M,0); cudaDeviceSynchronize();
        std::vector<int> g(mc); cudaMemcpy(g.data(),dc,mc*4,cudaMemcpyDeviceToHost);
        uint64_t bad=0; for(uint32_t z=0;z<c.M;++z)for(uint32_t y=0;y<c.M;++y)for(uint32_t x=0;x<c.M;++x)
            if(g[(size_t)x+c.M*((size_t)y+(size_t)c.M*z)]!=refCell(f,c.N,c.M,x,y,z)) bad++;
        printf("[pool N=%u M=%u] %s\n",c.N,c.M,bad?"FAIL":"OK"); if(bad) return 1;
        cudaFree(df); cudaFree(dc);
    }
    printf("voxel_pool_ref_test: ALL PASS\n"); return 0;
}
```

- [ ] **Step 2: Add header + CMake target (test fails to link)**

Create `lib/include/private/mimir/voxel_lod.hpp`:
```cpp
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace mimir {
// Reference max-pool of a fine N^3 int grid into a coarse M^3 grid (row-major x+N*(y+N*z)). Each coarse
// cell covers [c*N/M,(c+1)*N/M) per axis. NOT on the render path -- the vertex shader pools live; this
// mirrors that formula for tests and for an optional CA_LOD_CHECK. 0 < M <= N.
void voxelPoolMax(const int* fine, uint32_t N, int* coarse, uint32_t M, cudaStream_t stream);
}
```
Add to `lib/CMakeLists.txt` after the `lod_reduce_test` block:
```cmake
add_executable(voxel_pool_ref_test EXCLUDE_FROM_ALL tests/voxel_pool_ref_test.cu src/voxel_lod.cu)
target_include_directories(voxel_pool_ref_test PRIVATE include/private include/public)
target_link_libraries(voxel_pool_ref_test PRIVATE CUDA::cudart)
set_target_properties(voxel_pool_ref_test PROPERTIES CUDA_STANDARD 20 CUDA_ARCHITECTURES "${MIMIR_CUDA_ARCHITECTURES}")
```
Run: `./mimir-build-from-change.sh && cmake --build build --target voxel_pool_ref_test`
Expected: link error `undefined reference to mimir::voxelPoolMax`.

- [ ] **Step 3: Implement the reference kernel**

Create `lib/src/voxel_lod.cu`:
```cpp
#include "mimir/voxel_lod.hpp"
namespace mimir {
namespace { constexpr int kThreads = 256;
__global__ void poolMaxKernel(const int* fine, uint32_t N, int* coarse, uint32_t M){
    uint64_t mCells=(uint64_t)M*M*M;
    for(uint64_t c=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;c<mCells;c+=(uint64_t)blockDim.x*gridDim.x){
        uint32_t cx=(uint32_t)(c%M), cy=(uint32_t)((c/M)%M), cz=(uint32_t)(c/((uint64_t)M*M));
        uint32_t x0=(uint32_t)((uint64_t)cx*N/M),x1=(uint32_t)(((uint64_t)cx+1)*N/M);
        uint32_t y0=(uint32_t)((uint64_t)cy*N/M),y1=(uint32_t)(((uint64_t)cy+1)*N/M);
        uint32_t z0=(uint32_t)((uint64_t)cz*N/M),z1=(uint32_t)(((uint64_t)cz+1)*N/M);
        int m=0; for(uint32_t z=z0;z<z1;++z)for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x)
            m=max(m,fine[(uint64_t)x+N*((uint64_t)y+(uint64_t)N*z)]);
        coarse[c]=m;
    }
}}
void voxelPoolMax(const int* fine,uint32_t N,int* coarse,uint32_t M,cudaStream_t s){
    uint64_t b=((uint64_t)M*M*M+kThreads-1)/kThreads; if(b>2147483647ull)b=2147483647ull; if(b<1)b=1;
    poolMaxKernel<<<(uint32_t)b,kThreads,0,s>>>(fine,N,coarse,M);
}}
```
Run: `cmake --build build --target voxel_pool_ref_test && ./build/lib/voxel_pool_ref_test`
Expected: `[pool ...] OK` for every case, `voxel_pool_ref_test: ALL PASS`.

- [ ] **Step 4: Read the existing voxel vertex stage + grid mapping to match them**

Run: `cat shaders/voxel.slang | sed -n '1,25p'` and `grep -n "makeStructuredGrid" lib/src/engine.cpp`.
Record: (a) the exact `VertexData` struct the geometry stage consumes (`center`, `color`); (b) `makeStructuredGrid`'s per-axis world formula and its origin/spacing so the shader's `gridOrigin`/`gridSpacing` push constants reproduce it with M.

- [ ] **Step 5: Write `voxel_lod.slang` (new vertex stage; reuse voxel.slang's geometry+fragment)**

Create `shaders/voxel_lod.slang`:
```slang
import uniforms;

struct VertexData { float4 center : POSITION; float4 color : COLOR; };   // MUST match voxel.slang

struct LodPush {
    uint  fineN;       // fine grid resolution per axis
    uint  coarseM;     // coarse grid resolution per axis
    float4 gridOrigin;  // world origin (xyz); mirror makeStructuredGrid
    float4 gridSpacing; // world cell spacing at COARSE resolution (xyz)
};
[vk::push_constant] LodPush lod;

[vk::binding(0,1)] StructuredBuffer<int>    fineState;   // fine N^3 states, row-major x+N*(y+N*z)
[vk::binding(1,1)] StructuredBuffer<float4> colormap;    // state -> rgba

[shader("vertex")]
VertexData vertexLodMain(uint vid : SV_VertexID)
{
    uint M = lod.coarseM, N = lod.fineN;
    uint cx = vid % M, cy = (vid / M) % M, cz = vid / (M * M);
    // world center: mirror makeStructuredGrid at coarse resolution (REPLACE constants per Step 4).
    float3 center = lod.gridOrigin.xyz + (float3(cx, cy, cz) + 0.5) * lod.gridSpacing.xyz;
    // disjoint fine block [c*N/M,(c+1)*N/M) per axis; max-pool the state.
    uint x0 = cx * N / M, x1 = (cx + 1) * N / M;
    uint y0 = cy * N / M, y1 = (cy + 1) * N / M;
    uint z0 = cz * N / M, z1 = (cz + 1) * N / M;
    int m = 0;
    for (uint z = z0; z < z1; ++z)
      for (uint y = y0; y < y1; ++y)
        for (uint x = x0; x < x1; ++x)
            m = max(m, fineState[x + N * (y + N * z)]);
    VertexData o;
    o.center = float4(center, 1.0);
    o.color  = colormap[m];
    return o;
}
```
> Implementer note: keep `voxel.slang`'s `geometryMain2D/3D` + `fragmentMain` as the other stages of this pipeline (Task 2 wires the pipeline to compile `vertexLodMain` from this file plus the geometry/fragment entries from `voxel.slang`, mirroring how `pipeline.cpp` lists multiple entrypoints). The `VertexData` layout MUST be byte-identical to `voxel.slang`'s.

- [ ] **Step 6: Commit**

```bash
git add shaders/voxel_lod.slang lib/src/voxel_lod.cu lib/include/private/mimir/voxel_lod.hpp lib/tests/voxel_pool_ref_test.cu lib/CMakeLists.txt
git commit -m "shaders: voxel_lod vertex stage (in-shader M^3 max-pool) + pooling-formula parity test"
```

---

### Task 2: Build the Voxels-LOD graphics pipeline (procedural draw, SSBO reads, push constants)

**Files:**
- Modify: `lib/src/pipeline.cpp` (add a Voxels-LOD pipeline variant that uses `voxel_lod.slang`'s `vertexLodMain` + `voxel.slang`'s geometry/fragment, with NO vertex input bindings and a push-constant range + storage-buffer descriptor set)
- Modify: `lib/include/private/mimir/engine.hpp` (declare a `VoxelLodPipeline` holder + its push-constant struct)
- Modify: `lib/CMakeLists.txt` if a shader-copy/stamp list must include `voxel_lod.slang` (search for how `voxel.slang` is copied next to binaries; add `voxel_lod.slang` the same way)

**Interfaces:**
- Consumes: `shaders/voxel_lod.slang` (Task 1).
- Produces:
  ```cpp
  struct VoxelLodPush { uint32_t fineN, coarseM; float gridOrigin[4]; float gridSpacing[4]; };
  struct VoxelLodPipeline { VkPipeline pipeline; VkPipelineLayout layout; VkDescriptorSetLayout set_layout; };
  ```
  and a method `VoxelLodPipeline MimirInstance::makeVoxelLodPipeline(DomainType domain);` that builds the pipeline (empty vertex-input state, push-constant range = `sizeof(VoxelLodPush)`, one descriptor set with two `STORAGE_BUFFER` bindings for `fineState` and `colormap`).

- [ ] **Step 1: Read how a Voxels graphics pipeline is built today**

Run: `sed -n '30,60p' lib/src/pipeline.cpp` and the `ViewType::Voxels` case, plus how `pipeline.cpp` sets vertex-input state, push constants, and descriptor set layouts. Note the exact Slang compile struct (`module_path`, `entrypoints`) and the `VkPipelineVertexInputStateCreateInfo` used for Voxels.

- [ ] **Step 2: Declare the pipeline holder + push struct**

In `lib/include/private/mimir/engine.hpp`, add near the other pipeline/LOD members:
```cpp
    struct VoxelLodPush { uint32_t fineN, coarseM; float gridOrigin[4]; float gridSpacing[4]; };
    struct VoxelLodPipeline { VkPipeline pipeline = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE;
                              VkDescriptorSetLayout set_layout = VK_NULL_HANDLE; };
    VoxelLodPipeline makeVoxelLodPipeline(DomainType domain);
```

- [ ] **Step 3: Implement `makeVoxelLodPipeline`**

In `lib/src/pipeline.cpp` (or `engine.cpp` next to where graphics pipelines are built — match where the Voxels pipeline is created today), implement `makeVoxelLodPipeline`:
- Slang compile: `module_path = "shaders/voxel_lod.slang"` for the vertex entry `vertexLodMain`, and `module_path = "shaders/voxel.slang"` for `geometryMain{2D|3D}` + `fragmentMain`. Mirror the multi-entry compile the Voxels case already does (it lists `{"vertexMain", geom_entry, "fragmentMain"}`); here the vertex entry comes from a different module — follow the codebase's mechanism for a multi-module program (if the compiler only takes one module, add `vertexLodMain` INTO `voxel.slang` instead of a separate file, and adjust Task 1 to append the entry there).
- Vertex input: **empty** `VkPipelineVertexInputStateCreateInfo{ .vertexBindingDescriptionCount = 0, .vertexAttributeDescriptionCount = 0 }` (procedural draw).
- Topology: point list (same as the normal Voxels pipeline).
- Pipeline layout: one push-constant range `{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VoxelLodPush) }` + one descriptor set layout with two bindings `{0: STORAGE_BUFFER, 1: STORAGE_BUFFER}` at `VK_SHADER_STAGE_VERTEX_BIT`.
- Everything else (raster/depth/blend/MSAA/render pass): copy from the existing Voxels pipeline build so the coarse cubes render identically.

> Implementer note: if Slang cannot combine entrypoints from two modules in this codebase's build, the pragmatic fix is to **add `vertexLodMain` (and its `LodPush`/buffers) into `voxel.slang`** and specialize by entrypoint — revise Task 1 Step 5 accordingly. Prefer whichever the existing `make_pipeline`/`compileProgram` path supports; do not invent a new shader-build mechanism.

- [ ] **Step 4: Build to verify the pipeline compiles + the shader compiles**

Run: `./mimir-build-from-change.sh 2>&1 | grep -iE "error|slang|Built target mimir"`
Expected: `Built target mimir`, and the Slang compile of `voxel_lod.slang` succeeds (no shader errors). If the shader-copy step is needed, confirm `voxel_lod.slang` lands next to the binaries.

- [ ] **Step 5: Commit**

```bash
git add lib/src/pipeline.cpp lib/include/private/mimir/engine.hpp lib/CMakeLists.txt
git commit -m "engine: Voxels-LOD graphics pipeline (procedural M^3 draw, SSBO reads, push constants)"
```

---

### Task 3: Detect Voxels + `--lod`, bind the fine state/colormap SSBOs, and draw M³ coarse

**Files:**
- Modify: `lib/src/engine.cpp` (setup detection near the raster-LOD block; per-view bookkeeping; the draw path)
- Modify: `lib/include/private/mimir/engine.hpp` (per-view LOD record)

**Interfaces:**
- Consumes: `makeVoxelLodPipeline` + `VoxelLodPush` (Task 2), `makeStructuredGrid`'s mapping (Task 1 Step 4).
- Produces: for each qualifying Voxels view, the view renders M³ procedural coarse cubes via the LOD pipeline, reading the view's fine state + colormap SSBOs. No coarse buffers allocated.

- [ ] **Step 1: Add a per-view LOD record + storage**

In `engine.hpp`:
```cpp
    struct VoxelLodView {
        struct View* view = nullptr;
        VkBuffer     fine_state = VK_NULL_HANDLE; // fine N^3 state SSBO (view's color-index source)
        VkBuffer     colormap   = VK_NULL_HANDLE; // colormap SSBO (view's color source)
        VkDescriptorSet set = VK_NULL_HANDLE;
        VoxelLodPush push{};
    };
    std::vector<VoxelLodView> voxel_lod_views;
    VoxelLodPipeline voxel_lod_pipeline{};
```

- [ ] **Step 2: Detection + wiring at setup**

In `lib/src/engine.cpp`, after the raster-Markers LOD block and before the non-Markers warning, add:
```cpp
    if (options.pt_lod_cells > 0)
    {
        for (auto* view : views)
        {
            if (view->desc.type != ViewType::Voxels) continue;
            const uint32_t M = options.pt_lod_cells;
            const uint64_t fineCells = view->element_count;
            const uint32_t N = (uint32_t)llround(cbrt((double)fineCells));
            if (M == 0 || M >= N || (uint64_t)N*N*N != fineCells) continue;
            auto cit = view->desc.attributes.find(AttributeType::Color);
            if (cit == view->desc.attributes.end() || !hasIndexing(cit->second)
                || cit->second.indexing.index_size != (int)sizeof(int)) continue;

            if (voxel_lod_pipeline.pipeline == VK_NULL_HANDLE)
                voxel_lod_pipeline = makeVoxelLodPipeline(view->desc.domain);

            // fine state SSBO = the color-index source buffer; colormap SSBO = the color source buffer.
            VkBuffer fine_state = getVulkanBuffer(cit->second.indexing.source);
            VkBuffer colormap   = getVulkanBuffer(cit->second.source);
            VkDescriptorSet set = allocVoxelLodDescriptor(fine_state, colormap); // 2 STORAGE_BUFFERs

            // push constants: mirror makeStructuredGrid's origin/spacing at coarse M (Task 1 Step 4).
            VoxelLodPush push{}; push.fineN = N; push.coarseM = M;
            fillVoxelLodGridMapping(push, M);  // sets gridOrigin/gridSpacing from makeStructuredGrid's formula

            // Repoint the view to the LOD pipeline + procedural M^3 draw (NO vertex buffers).
            view->pipeline = voxel_lod_pipeline.pipeline;
            view->vb_count = 0;
            view->draw_count = (uint32_t)((uint64_t)M*M*M);   // M^3 < 2^32 for M <= 1625
            view->element_count = (uint64_t)M*M*M;
            view->instance_count = 1;

            voxel_lod_views.push_back(VoxelLodView{ view, fine_state, colormap, set, push });
            spdlog::info("Voxels LOD (in-shader): {}^3 -> {}^3 grid ({} -> {} cubes, {:.1f}x fewer), max-pooled",
                (unsigned)N, (unsigned)M, (unsigned long long)fineCells,
                (unsigned long long)((uint64_t)M*M*M), (double)fineCells/((double)M*M*M));
        }
    }
```
> Implementer notes: `getVulkanBuffer(AllocHandle)` / `getMemoryVulkan` already exist (used by createView's indexing path, `engine.cpp:1466`); use the codebase's actual accessor names. `allocVoxelLodDescriptor` and `fillVoxelLodGridMapping` are small helpers to add in the same file (descriptor from `voxel_lod_pipeline.set_layout`; mapping copied verbatim from `makeStructuredGrid`).

- [ ] **Step 3: Bind the descriptor + push constants + issue the procedural draw**

Find where each view is drawn (the `drawChunk`/`drawElements` path, `engine.cpp:2576-2660`). For a view in `voxel_lod_views` (or simply `view->vb_count == 0 && view->pipeline == voxel_lod_pipeline.pipeline`): after binding the pipeline, `vkCmdPushConstants(cmd, voxel_lod_pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(VoxelLodPush), &rec.push)`, `vkCmdBindDescriptorSets(... set 1 ..., rec.set ...)`, then `vkCmdDraw(cmd, view->draw_count, 1, 0, 0)` with NO `vkCmdBindVertexBuffers`. Keep the normal MVP/uniform descriptor (set 0) bound as usual.

- [ ] **Step 4: Guard the non-Markers warning**

Update the earlier `--lod ignored` warning condition to also require `voxel_lod_views.empty()`:
```cpp
    if (options.pt_lod_cells > 0 && !rt_enabled && !lod_context.active()
        && voxel_lod_views.empty() && supportsRayTracing(physical_device.handle))
```

- [ ] **Step 5: Build**

Run: `./mimir-build-from-change.sh 2>&1 | grep -iE "error|Built target mimir"`
Expected: `Built target mimir`.

- [ ] **Step 6: Commit**

```bash
git add lib/src/engine.cpp lib/include/private/mimir/engine.hpp
git commit -m "engine: wire Voxels --lod to the in-shader coarse pipeline (SSBO binds + M^3 draw)"
```

---

### Task 4: `--lod` in the CA3D-voxels sample + end-to-end verification

**Files:**
- Modify: `samples/CA3D-voxels/main.cu`

**Interfaces:**
- Consumes: the automatic library behavior (Tasks 2-3).

- [ ] **Step 1: Switch to the options overload + parse `--lod M`**

In `samples/CA3D-voxels/main.cu`, relax the `argc` guard to allow `--lod M`, then:
```cpp
    unsigned int lod_cells = 0;
    for (int i = 1; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--lod") lod_cells = (unsigned)std::stoul(argv[i+1]);
    ViewerOptions opts;                     // read the correct field names from options.hpp
    opts.window.size = {width, height};
    opts.pt_lod_cells = lod_cells;
    createInstance(opts, &instance);
```
> Implementer note: confirm the `ViewerOptions` window-size field name in `lib/include/public/mimir/options.hpp` and use it exactly.

- [ ] **Step 2: Build the sample**

Run: `cmake --build samples/CA3D-voxels/build 2>&1 | grep -iE "error|Built target"` (or the repo's sample-build script if it wires CA3D-voxels).
Expected: builds cleanly.

- [ ] **Step 3: Smoke test — coarse log line + no validation errors**

Run (n=64, GPU, exit after 1 step by feeding EOF to the `getchar()` prompts):
```bash
cd samples/CA3D-voxels/build && echo | ./CA3D-voxels 64 1 8 1 1 0.3 1 --lod 16 2>&1 | grep -iE "Voxels LOD|VUID|error|abort" | head
```
Expected: `Voxels LOD (in-shader): 64^3 -> 16^3 grid (262144 -> 4096 cubes, 64.0x fewer), max-pooled` and NO Vulkan validation (`VUID`) or error lines.

- [ ] **Step 4: Correctness gate against the CPU reference (`CA_LOD_CHECK`)**

Add to the sample, guarded by `getenv("CA_LOD_CHECK")`, a one-shot after the first sim step: copy the fine state to host, compute the CPU max-pool reference (same formula as `voxel_pool_ref_test`), AND compute the GPU reference by calling `mimir::voxelPoolMax` (from `mimir/voxel_lod.hpp`) on the same fine state into a scratch buffer; assert they match and print `CA_LOD_CHECK: OK/FAIL`. This validates the exact pooling formula the shader implements end-to-end on live sim data. Run:
```bash
CA_LOD_CHECK=1 echo | ./CA3D-voxels 64 1 8 1 1 0.3 1 --lod 16 2>&1 | grep CA_LOD_CHECK
```
Expected: `CA_LOD_CHECK: OK`.
> Implementer note: include `"mimir/voxel_lod.hpp"` in the sample and link against the mimir lib (already linked). This reuses the tested reference kernel; the shader uses the identical formula, so a passing gate + the clean smoke test together establish end-to-end correctness.

- [ ] **Step 5: Visual confirmation (manual)**

Run interactively `./CA3D-voxels 128 1 8 1 5 0.3 1 --lod 32` and confirm the rendered cube cloud is a coarsened (blockier) version of the full run, updating each step. This is a manual check; note the observation in the commit message.

- [ ] **Step 6: Commit**

```bash
git add samples/CA3D-voxels/main.cu
git commit -m "CA3D-voxels: --lod M demo + CA_LOD_CHECK gate for the in-shader Voxels LOD"
```

---

### Task 5: Document Voxels semantics on the LOD option

**Files:**
- Modify: `lib/include/public/mimir/options.hpp`

- [ ] **Step 1: Extend the `pt_lod_cells` comment**

Append near line 185:
```cpp
    // For ViewType::Voxels this is the coarse grid resolution M: the fine N^3 int-state grid is
    // max-pooled ON THE FLY in the vertex shader into an M^3 grid and M^3 cubes are drawn (M < N; M >= N
    // is a no-op). No coarse data is materialized. State must be int32 (the colormap index type). For
    // ViewType::Markers it is the point-cloud reduction grid. Other view types ignore it (warned).
```

- [ ] **Step 2: Build + commit**

Run: `./mimir-build-from-change.sh 2>&1 | grep -iE "error|Built target mimir"` (expect `Built target mimir`), then:
```bash
git add lib/include/public/mimir/options.hpp
git commit -m "options: document in-shader Voxels grid-coarsening semantics for pt_lod_cells"
```

---

## Self-Review

**Spec coverage:** in-shader pooling (no coarse copy) → Task 1 shader + Task 2 pipeline + Task 3 draw. Automatic via `--lod` → Task 3. No compute kernel/coarse buffers/extra sync → architecture (reference kernel exists only for tests/verify, Task 1). Ping/pong → each Voxels view independently repointed to the LOD pipeline with its own fine-state SSBO; visibility untouched (Task 3 loops all Voxels views). Demo + verify → Task 4. Docs → Task 5. ✓

**Placeholder scan:** The remaining "read the exact X from source" notes (grid mapping in Task 1/3; `ViewerOptions` field in Task 4; accessor/entrypoint mechanism in Task 2/3) are deliberate DRY instructions naming the exact file/function to copy from — the alternative (guessing Vulkan/Slang specifics) is worse. Task 2 Step 3 explicitly provides the fallback (fold `vertexLodMain` into `voxel.slang`) if the build can't combine two shader modules — that is a real decision point, resolved, not a gap.

**Type consistency:** `VoxelLodPush{fineN,coarseM,gridOrigin[4],gridSpacing[4]}` matches the shader's `LodPush{fineN,coarseM,gridOrigin,gridSpacing}` and is used identically in Tasks 2-3. `voxelPoolMax(const int*,uint32_t,int*,uint32_t,cudaStream_t)` used identically in Tasks 1, 4. `VoxelLodView`/`voxel_lod_views`/`voxel_lod_pipeline` defined Task 3, consumed in the same task's draw step. Consistent.

**Biggest risk to flag at execution:** whether the Slang build in this repo can compile a program whose vertex entry lives in `voxel_lod.slang` while geometry/fragment live in `voxel.slang`. Task 2 Step 3 resolves it up front: if not, move `vertexLodMain` into `voxel.slang`. Decide this in Task 2 before writing pipeline code, so Tasks 1/3 reference the right module. Secondary risk: the fine-state buffer must carry `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` — verify the interop/attribute buffer already has it (createView's indexing buffers use `vb_usage`); if not, the LOD path must request that usage when the source is an indexed Voxels color (small createView tweak, add as a Task 3 sub-step if validation flags a missing usage bit).
