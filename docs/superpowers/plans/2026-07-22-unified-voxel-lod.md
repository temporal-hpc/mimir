# Unified Voxel-LOD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make voxels the default LOD representation across all lit light models (phong, phong-mesh, path-tracing; `none` excluded), replacing the PT-only `--lod-voxel` flag with `--lod-shape sphere|voxel` (default voxel).

**Architecture:** PT already voxelizes via its box branch — Task 1 just flips the default and swaps the flag. For raster, we reuse the existing instanced-mesh (SphereMesh) path: Task 2 adds a per-vertex normal attribute to that path (a no-op refactor for the icosphere), and Task 3 adds a unit-cube template plus a single override at `engine.cpp:1508` that forces voxel-LOD lit views to the SphereMesh render mode — which makes the whole existing mesh pipeline/shader/draw path render cubes automatically.

**Tech Stack:** C++17, CUDA, Vulkan, Slang shaders. CMake (two trees: `build/` for `libmimir.a`; `samples/remote-rendering/build/` for `rr-server`/`rr-client`). Shaders are copied next to the binary and compiled at runtime by Slang.

## Global Constraints

- **Default is voxel:** `ViewerOptions::lod_voxel` defaults `true`; voxel applies only under `--lod` and only to lit models (`light_model != None`).
- **`none` never voxelizes** (flat pixel points — no surface).
- **Full-cell tiling:** cube half-extent = `LodContext::voxelHalfExtent()` (= `1/grid_n`); cell-center placement is auto-forced whenever voxels render.
- **Do not touch the CA3D `voxel_boxes` pipeline** or the LOD reduction algorithm.
- **Build order** (after editing `lib/` or shaders):
  ```bash
  touch lib/src/remote.cpp                                  # defeat clock-skew skips
  cmake --build build --target mimir -j
  cmake --build samples/remote-rendering/build -j           # relinks rr-server AND re-copies shaders
  ```
  (Building the sample's default target — not just `--target rr-server` — is required after a shader edit, so `rr-server-shaders` re-copies the `.slang` files next to the binary.)
- **Verification is integration-style** (background server + headless `rr-client` + log grep + saved `.ppm` → `.png` via `python3 -c "from PIL import Image; ..."`). No unit-test harness exists for the render pipeline. A 640x480 / 20000-particle / `--k 6` scene renders fast and shows shape clearly.

---

## File Structure

- `lib/include/public/mimir/options.hpp` — `pt_lod_voxel` → `lod_voxel = true`.
- `lib/include/private/mimir/engine.hpp` — declare `ensureCubeMesh()` + `cube_vbo`/`cube_ibo`/`cube_index_count`.
- `lib/src/engine.cpp` — force cell-center in `prepare()`; PT plumbing rename; render_mode override at createView; cube template selection + interleaved stride in mesh view setup; `marker_size` voxel branch; `ensureCubeMesh`; make `ensureSphereMesh` interleave a normal.
- `lib/src/pipeline.cpp` — mesh vertex-input: binding-0 stride 24, add location-2 normal attribute.
- `shaders/marker_mesh.slang` — read explicit normal (location 2), raw-offset instead of `normalize(in_local)`.
- `samples/remote-rendering/rr-server.cu` — remove `--lod-voxel`; add `--lod-shape sphere|voxel`.
- PT path (`raytracing.cpp`, `pathtrace.slang`) — unchanged (already implemented).

---

## Task 1: Option rename + `--lod-shape` CLI + PT default-voxel

Flips the default to voxel and swaps the flag. PT renders voxels by default; raster is unchanged (still spheres — wired in Task 3). Independently reviewable: PT voxel-by-default works, `--lod-shape sphere` reverts to PT spheres, `--lod-voxel` is gone.

**Files:**
- Modify: `lib/include/public/mimir/options.hpp:238`
- Modify: `lib/src/engine.cpp:238` (PT plumbing) and `lib/src/engine.cpp:215` (`prepare()` start, cell-center force)
- Modify: `samples/remote-rendering/rr-server.cu` (remove `--lod-voxel`, add `--lod-shape`)

**Interfaces:**
- Produces: `ViewerOptions::lod_voxel` (bool, default `true`) — consumed by Task 3's raster path and the PT plumbing here.

- [ ] **Step 1: Rename the option, default true**

In `lib/include/public/mimir/options.hpp`, replace the `pt_lod_voxel` field (currently `:238`):
```cpp
    bool pt_lod_voxel = false;
```
with:
```cpp
    // Render LOD representatives as solid grid-aligned cubes (voxels) instead of spheres. Default:
    // true. Applies only under pt_lod_cells > 0 and only to lit models (phong / phong-mesh /
    // path-tracing); `none` (flat points) ignores it. Forces cell-center placement + full-cell fill.
    // Set false (rr-server --lod-shape sphere) to render LOD as spheres and honour lod_centroid.
    bool lod_voxel = true;
```

- [ ] **Step 2: Update the PT plumbing**

In `lib/src/engine.cpp` (currently `:238`), change:
```cpp
                raytracing.lod_voxel = options.pt_lod_voxel;
```
to:
```cpp
                raytracing.lod_voxel = options.lod_voxel;
```

- [ ] **Step 3: Force cell-center for voxel LOD in `prepare()`**

In `lib/src/engine.cpp`, at the very start of `MimirInstance::prepare()` body (currently `:215`, immediately inside the opening brace), add:
```cpp
    // Voxel LOD tiles only on the grid lattice, so cell-center placement is mandatory whenever voxels
    // render (all lit models). Force it here -- before both the PT (bindScene) and raster LOD inits
    // read options.lod_centroid -- so every caller (not just rr-server) is consistent. `none` and the
    // sphere opt-out (lod_voxel=false) keep whatever placement was requested.
    if (options.lod_voxel && options.pt_lod_cells > 0 && options.light_model != LightModel::None)
    {
        if (options.lod_centroid)
        {
            spdlog::info("LOD voxels: forcing cell-center placement (centroid is off-lattice, breaks tiling)");
        }
        options.lod_centroid = false;
    }
```

- [ ] **Step 4: Remove the `--lod-voxel` flag and its plumbing from rr-server**

In `samples/remote-rendering/rr-server.cu`, delete these four pieces:

1. The declaration (currently `:233`):
```cpp
    bool lod_voxel_render   = false;  // --lod-voxel: draw LOD reps as full-cell cubes (forces cell-center)
```
2. The flag parse (currently `:253`):
```cpp
        if (a == "--lod-voxel") { lod_voxel_render = true; continue; } // LOD reps as full-cell cubes (flag)
```
3. The entire force/warn block (currently `:437`-`:459`, the comment starting `// LOD voxel mode is a path-tracing-only feature ...` through its closing brace) — remove it whole. (Cell-center forcing now lives in the engine, Step 3.)
4. The option assignment (currently `:461`):
```cpp
    options.pt_lod_voxel         = lod_voxel_render;
```

- [ ] **Step 5: Add `--lod-shape` declaration + parse**

In `samples/remote-rendering/rr-server.cu`, next to the other LOD declarations (near `bool lod_centroid = true;`), add:
```cpp
    bool lod_voxel_shape    = true;   // --lod-shape voxel|sphere: LOD as cubes (default) vs spheres
```

In the value-consuming parse group (next to `--lod-placement`, currently near `:258`), add:
```cpp
            else if (a == "--lod-shape") {
                if      (v == "voxel")  lod_voxel_shape = true;
                else if (v == "sphere") lod_voxel_shape = false;
                else { fprintf(stderr, "Unknown --lod-shape '%s' (use voxel|sphere)\n", v.c_str()); return EXIT_FAILURE; }
            }
```

- [ ] **Step 6: Assign into options + keep the voxel notes**

In `samples/remote-rendering/rr-server.cu`, where the LOD options are assigned (near `options.lod_centroid = lod_centroid;`), add:
```cpp
    options.lod_voxel         = lod_voxel_shape;
    if (lod_voxel_shape && lod_cells > 0 && light_model != LightModel::None && size_set)
    {
        fprintf(stdout, "rr-server: LOD voxels ignore --size; cubes always fill the cell\n");
    }
```
(The old `options.pt_lod_cells = lod_cells;` / `options.lod_centroid = lod_centroid;` lines stay as they are.)

- [ ] **Step 7: Add `--lod-shape` help text; remove `--lod-voxel` help**

In `samples/remote-rendering/rr-server.cu`, delete the four `--lod-voxel` help lines (the block starting `"  --lod-voxel        Render each LOD representative ..."`), and in their place add:
```cpp
        "  --lod-shape S      LOD representative shape: voxel (default) = solid grid-aligned cubes\n"
        "                     (forces cell-center placement, ignores --size for extent); sphere =\n"
        "                     round spheres honouring --lod-placement and --size. Lit models only;\n"
        "                     --light-model none always draws flat points. No-op without --lod.\n"
```

- [ ] **Step 8: Build**

Run:
```bash
touch lib/src/remote.cpp
cmake --build build --target mimir -j 2>&1 | grep -iE "error|Built target mimir" | tail -3
cmake --build samples/remote-rendering/build --target rr-server -j 2>&1 | grep -iE "error|Built target rr-server" | tail -3
```
Expected: both build with no `error:` lines. (No `pt_lod_voxel` references remain — verify with `grep -rn pt_lod_voxel lib samples`, expect no hits.)

- [ ] **Step 9: Verify PT voxel-by-default and the sphere opt-out**

```bash
SCR=/tmp/uvl-t1; mkdir -p $SCR
# default (no shape flag) -> voxels
SP=$SCR/pt-default.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 60 \
  --light-model path-tracing --lod 64 --spp 1 --fps 60 > "$SP" 2>&1 &
PID=$!; for i in $(seq 1 30); do grep -q "listening on port" "$SP" && break; sleep 1; done
grep -q "voxel half-extent" "$SP" && echo "PASS: PT voxel by default" || { echo "FAIL"; grep "LOD emitted" "$SP"; }
kill $PID 2>/dev/null
# opt out -> spheres
SP2=$SCR/pt-sphere.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 60 \
  --light-model path-tracing --lod 64 --lod-shape sphere --spp 1 --fps 60 > "$SP2" 2>&1 &
PID=$!; for i in $(seq 1 30); do grep -q "listening on port" "$SP2" && break; sleep 1; done
grep -q "sphere radius" "$SP2" && echo "PASS: --lod-shape sphere -> spheres" || echo "FAIL: sphere opt-out"
kill $PID 2>/dev/null
```
Expected: both `PASS`.

- [ ] **Step 10: Commit**

```bash
git add lib/include/public/mimir/options.hpp lib/src/engine.cpp samples/remote-rendering/rr-server.cu
git commit -m "feat(rr): voxel LOD by default for PT; --lod-shape replaces --lod-voxel

Rename ViewerOptions::pt_lod_voxel -> lod_voxel (default true) and drive PT
voxel rendering by default. Replace rr-server --lod-voxel with --lod-shape
sphere|voxel (default voxel). Force cell-center for voxel LOD in prepare()
so every caller is consistent. Raster still renders spheres (wired next).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W"
```

---

## Task 2: Add a per-vertex normal to the instanced-mesh path (no-op refactor)

The cube needs per-face normals, but the mesh path currently derives the normal from the vertex position (`normalize(in_local)`), which is only valid for a unit sphere. Add an explicit normal attribute now, with the icosphere setting `normal = position`, so this task changes nothing visually — it's the safe refactor that Task 3 builds on. Independently reviewable: phong-mesh renders identically before/after.

**Files:**
- Modify: `lib/src/engine.cpp` — `ensureSphereMesh` (interleave normal), mesh view-setup stride (`:1746`)
- Modify: `lib/src/pipeline.cpp:396-405` — vertex input
- Modify: `shaders/marker_mesh.slang:22-34` — vertex shader

**Interfaces:**
- Produces: instanced-mesh template VBO layout is now interleaved `{vec3 position; vec3 normal;}` (stride 24), binding-0 attributes: location 0 = position (offset 0), location 2 = normal (offset 12). Task 3's cube template writes this same layout.

- [ ] **Step 1: Interleave a normal into the icosphere VBO**

In `lib/src/engine.cpp`, in `ensureSphereMesh` (currently `:1094`), replace the VBO sizing + upload so vertices carry `{position, normal=position}`. Change the `vsize` computation and the vbo memcpy. Find:
```cpp
    VkDeviceSize vsize = verts.size() * sizeof(glm::vec3);
```
change to:
```cpp
    // Interleaved {position, normal}. For the unit icosphere the normal IS the position (unit sphere),
    // matching the shared instanced-mesh layout the cube template also uses (Task 3).
    struct MeshVertex { glm::vec3 pos; glm::vec3 nrm; };
    std::vector<MeshVertex> mesh_verts; mesh_verts.reserve(verts.size());
    for (const auto& v : verts) { mesh_verts.push_back({ v, v }); }
    VkDeviceSize vsize = mesh_verts.size() * sizeof(MeshVertex);
```
Then find the vbo upload:
```cpp
    vkMapMemory(device, vbo_mem, 0, vsize, 0, &p); std::memcpy(p, verts.data(), vsize);
```
change to:
```cpp
    vkMapMemory(device, vbo_mem, 0, vsize, 0, &p); std::memcpy(p, mesh_verts.data(), vsize);
```

- [ ] **Step 2: Update the mesh view-setup stride**

In `lib/src/engine.cpp`, in the SphereMesh view setup (currently `:1746`), change:
```cpp
        view.vbo_stride[0] = sizeof(glm::vec3); view.vbo_rate[0] = VK_VERTEX_INPUT_RATE_VERTEX;
```
to:
```cpp
        view.vbo_stride[0] = 2 * sizeof(glm::vec3); view.vbo_rate[0] = VK_VERTEX_INPUT_RATE_VERTEX; // {pos, normal}
```

- [ ] **Step 3: Update the pipeline vertex input**

In `lib/src/pipeline.cpp` (currently `:396-405`), replace the SphereMesh binding/attribute block:
```cpp
        constexpr uint32_t vec3_stride = static_cast<uint32_t>(sizeof(glm::vec3));
        vert.binding = {
            { .binding = 0, .stride = vec3_stride, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
            { .binding = 1, .stride = vec3_stride, .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE },
        };
        vert.attribute = {
            { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
            { .location = 1, .binding = 1, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
        };
        return vert;
```
with (binding 0 now interleaves position + normal; location 2 = normal at offset 12):
```cpp
        constexpr uint32_t vec3_stride  = static_cast<uint32_t>(sizeof(glm::vec3));
        constexpr uint32_t vertex_stride = 2u * vec3_stride; // {position, normal}
        vert.binding = {
            { .binding = 0, .stride = vertex_stride, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
            { .binding = 1, .stride = vec3_stride,   .inputRate = VK_VERTEX_INPUT_RATE_INSTANCE },
        };
        vert.attribute = {
            { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
            { .location = 1, .binding = 1, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 0 },
            { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = vec3_stride },
        };
        return vert;
```

- [ ] **Step 4: Read the explicit normal in the mesh vertex shader**

In `shaders/marker_mesh.slang`, replace `vertexMain` (currently `:21-35`):
```slang
[shader("vertex")]
VertexOutput vertexMain(
    [[vk::location(0)]] float3 in_local  : POSITION,   // template icosphere vertex (per-vertex)
    [[vk::location(1)]] float3 in_center : TEXCOORD0)  // particle center       (per-instance)
{
    float4 center_view = mul(mul(float4(in_center, 1.0), mvp.model), mvp.view);
    float3 n = normalize(in_local);
    float3 vpos = center_view.xyz + n * view.default_size;

    VertexOutput output;
    output.pos     = mul(float4(vpos, 1.0), mvp.proj);
    output.vpos    = vpos;
    output.vnormal = n;
    return output;
}
```
with (raw local offset + explicit normal; identical result for the unit icosphere, correct for the cube):
```slang
[shader("vertex")]
VertexOutput vertexMain(
    [[vk::location(0)]] float3 in_local  : POSITION,   // template vertex, local space (per-vertex)
    [[vk::location(1)]] float3 in_center : TEXCOORD0,  // particle/cell center     (per-instance)
    [[vk::location(2)]] float3 in_normal : NORMAL)     // template vertex normal   (per-vertex)
{
    float4 center_view = mul(mul(float4(in_center, 1.0), mvp.model), mvp.view);
    // Offset the template vertex (already unit for the sphere; +/-1 corner for the cube) by the marker
    // size; the normal is supplied per-vertex (sphere: == position; cube: per-face), not derived. The
    // marker model is identity in this sample, so the view-space normal matches the previous
    // normalize(in_local) for the sphere; the cube gets true flat per-face normals.
    float3 vpos = center_view.xyz + in_local * view.default_size;
    VertexOutput output;
    output.pos     = mul(float4(vpos, 1.0), mvp.proj);
    output.vpos    = vpos;
    output.vnormal = normalize(in_normal);
    return output;
}
```

- [ ] **Step 5: Build**

```bash
touch lib/src/remote.cpp
cmake --build build --target mimir -j 2>&1 | grep -iE "error|Built target mimir" | tail -3
cmake --build samples/remote-rendering/build -j 2>&1 | grep -iE "error|Built target rr-server|rr-server-shaders" | tail -4
```
Expected: builds; `rr-server-shaders` re-copies. Confirm the copied shader has the normal input:
```bash
grep -c "in_normal" samples/remote-rendering/build/shaders/marker_mesh.slang
```
Expected: `>= 1`.

- [ ] **Step 6: Verify phong-mesh is visually unchanged (no-op refactor)**

```bash
SCR=/tmp/uvl-t2; mkdir -p $SCR
SP=$SCR/phongmesh.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000 413111 300 \
  --light-model phong-mesh --size 9 --k 6 --fps 60 > "$SP" 2>&1 &
PID=$!; for i in $(seq 1 30); do grep -q "listening on port" "$SP" && break; sleep 1; done
timeout 40 samples/remote-rendering/build/rr-client 127.0.0.1 9000 "" tcp 30 2>&1 | tail -1
cp rr-client.ppm $SCR/phongmesh.ppm
python3 -c "from PIL import Image; Image.open('$SCR/phongmesh.ppm').save('$SCR/phongmesh.png')"
kill $PID 2>/dev/null
echo "inspect $SCR/phongmesh.png -- must still show round lit icospheres"
```
Expected: `received 30 frames`; the PNG shows the same round lit icospheres as before (no facets/artifacts). Open it to confirm.

- [ ] **Step 7: Commit**

```bash
git add lib/src/engine.cpp lib/src/pipeline.cpp shaders/marker_mesh.slang
git commit -m "refactor(mesh): explicit per-vertex normal in the instanced-mesh path

Add a per-vertex normal attribute (location 2) to the instanced marker-mesh
vertex input; the icosphere template sets normal = position, so this is a
visual no-op. The vertex shader now offsets by the raw template vertex and
reads the supplied normal instead of normalize(in_local) -- prep for the cube
template (Task 3), which needs per-face normals a unit-sphere trick can't give.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W"
```

---

## Task 3: Cube template + route lit voxel-LOD through the mesh path

Adds the unit-cube template and the single render_mode override that makes voxel-LOD lit views render as instanced cubes. Independently reviewable: phong / phong-mesh / PT all render cubes by default under `--lod`; `--lod-shape sphere` reverts each to spheres; `none` unaffected.

**Files:**
- Modify: `lib/include/private/mimir/engine.hpp` — declare `ensureCubeMesh()` + cube buffers/index count
- Modify: `lib/src/engine.cpp` — `ensureCubeMesh`; render_mode override (`:1508`); cube template in mesh setup (`:1737`); `marker_size` voxel branch (`:3263`)

**Interfaces:**
- Consumes: `ViewerOptions::lod_voxel` (Task 1), the interleaved `{pos,normal}` mesh layout (Task 2), `LodContext::voxelHalfExtent()` (already exists).

- [ ] **Step 1: Declare the cube template members**

In `lib/include/private/mimir/engine.hpp`, next to the `ensureSphereMesh` / `sphere_vbo` declarations (currently `:271-274`), add:
```cpp
    // Shared template unit cube for voxel LOD (instanced like the icosphere; 24 verts w/ per-face
    // normals, 36 indices). Built lazily the first time a voxel-LOD lit mesh view is set up.
    VkBuffer cube_vbo = VK_NULL_HANDLE;   // interleaved {position, normal}, 24 vertices
    VkBuffer cube_ibo = VK_NULL_HANDLE;   // 36 uint32 indices (12 triangles)
    uint32_t cube_index_count = 0;
    void ensureCubeMesh(); // build cube_vbo/ibo once
```

- [ ] **Step 2: Implement `ensureCubeMesh`**

In `lib/src/engine.cpp`, immediately after `ensureSphereMesh`'s closing brace (currently `:1163`), add:
```cpp
void MimirInstance::ensureCubeMesh()
{
    if (cube_index_count > 0) { return; } // already built

    struct MeshVertex { glm::vec3 pos; glm::vec3 nrm; };
    // Unit cube [-1,1]^3, 4 vertices per face so each face carries a flat outward normal. Vertex order
    // per face is CCW when viewed from outside (matches the icosphere's outward winding).
    const glm::vec3 faces[6][4] = {
        {{ 1,-1,-1},{ 1, 1,-1},{ 1, 1, 1},{ 1,-1, 1}}, // +X
        {{-1,-1, 1},{-1, 1, 1},{-1, 1,-1},{-1,-1,-1}}, // -X
        {{-1, 1,-1},{-1, 1, 1},{ 1, 1, 1},{ 1, 1,-1}}, // +Y
        {{-1,-1, 1},{-1,-1,-1},{ 1,-1,-1},{ 1,-1, 1}}, // -Y
        {{-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}}, // +Z
        {{ 1,-1,-1},{-1,-1,-1},{-1, 1,-1},{ 1, 1,-1}}, // -Z
    };
    const glm::vec3 normals[6] = {
        { 1,0,0},{-1,0,0},{0, 1,0},{0,-1,0},{0,0, 1},{0,0,-1},
    };
    std::vector<MeshVertex> verts; verts.reserve(24);
    std::vector<uint32_t>   indices; indices.reserve(36);
    for (int f = 0; f < 6; ++f)
    {
        uint32_t base = static_cast<uint32_t>(verts.size());
        for (int v = 0; v < 4; ++v) { verts.push_back({ faces[f][v], normals[f] }); }
        indices.insert(indices.end(), { base+0, base+1, base+2, base+0, base+2, base+3 });
    }

    VkDeviceSize vsize = verts.size() * sizeof(MeshVertex);
    VkDeviceSize isize = indices.size() * sizeof(uint32_t);
    auto flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    auto available = physical_device.memory.memoryProperties;
    VkMemoryRequirements memreq{};

    cube_vbo = createBuffer(device, vsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    vkGetBufferMemoryRequirements(device, cube_vbo, &memreq);
    auto vbo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, cube_vbo, vbo_mem, 0));

    cube_ibo = createBuffer(device, isize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    vkGetBufferMemoryRequirements(device, cube_ibo, &memreq);
    auto ibo_mem = allocateMemory(device, available, memreq, flags);
    validation::checkVulkan(vkBindBufferMemory(device, cube_ibo, ibo_mem, 0));

    void* p = nullptr;
    vkMapMemory(device, vbo_mem, 0, vsize, 0, &p); std::memcpy(p, verts.data(), vsize);
    vkUnmapMemory(device, vbo_mem);
    vkMapMemory(device, ibo_mem, 0, isize, 0, &p); std::memcpy(p, indices.data(), isize);
    vkUnmapMemory(device, ibo_mem);

    cube_index_count = static_cast<uint32_t>(indices.size());
    VkBuffer vbo = cube_vbo, ibo = cube_ibo;
    deletors.views.add([=,this]{
        vkFreeMemory(device, vbo_mem, nullptr); vkDestroyBuffer(device, vbo, nullptr);
        vkFreeMemory(device, ibo_mem, nullptr); vkDestroyBuffer(device, ibo, nullptr);
    });
    spdlog::info("Voxel LOD: unit cube template built ({} tris) for instanced raster", cube_index_count / 3);
}
```

- [ ] **Step 3: Override render_mode to SphereMesh for voxel-LOD lit views**

In `lib/src/engine.cpp`, immediately after the `switch (options.light_model)` block in createView that sets `marker_opts.render_mode` (currently ends `:1540`, just before the closing brace of the `if (view.desc.type == ViewType::Markers ...)`), add:
```cpp
        // Voxel LOD (lit models) renders as an instanced cube mesh: route the view through the
        // SphereMesh machinery regardless of the light model's own render mode (phong's Sphere3D
        // impostor included). The cube-vs-icosphere template is chosen in the mesh view setup below.
        // `none` (Flat2D) and the sphere opt-out (lod_voxel=false) are untouched. PathTracing keeps
        // its raster fallback mode; the RT path ignores render_mode anyway.
        if (options.lod_voxel && options.pt_lod_cells > 0
            && options.light_model != LightModel::None
            && options.light_model != LightModel::PathTracing)
        {
            marker_opts.render_mode = MarkerOptions::RenderMode::SphereMesh;
        }
```

- [ ] **Step 4: Select the cube template in the mesh view setup**

In `lib/src/engine.cpp`, in the SphereMesh view setup (currently `:1737`), replace:
```cpp
        ensureSphereMesh();
        VkBuffer instance_positions = view.vbo[0]; // interop particle centers (per-instance)
        uint64_t particle_count     = view.element_count;
        view.vbo[0]        = sphere_vbo;           // binding 0: unit icosphere vertices
```
with:
```cpp
        // Voxel LOD lit views use the cube template; everything else uses the icosphere. Both share the
        // interleaved {pos, normal} layout and the same instanced pipeline/shader.
        const bool use_cube = options.lod_voxel && options.pt_lod_cells > 0
            && options.light_model != LightModel::None;
        VkBuffer  tmpl_vbo   = use_cube ? (ensureCubeMesh(),   cube_vbo)         : (ensureSphereMesh(), sphere_vbo);
        VkBuffer  tmpl_ibo   = use_cube ? cube_ibo                               : sphere_ibo;
        uint32_t  tmpl_count = use_cube ? cube_index_count                        : sphere_index_count;
        VkBuffer instance_positions = view.vbo[0]; // interop particle centers (per-instance)
        uint64_t particle_count     = view.element_count;
        view.vbo[0]        = tmpl_vbo;             // binding 0: template vertices ({pos, normal})
```
Then, further down in the same block, replace:
```cpp
        view.ibo           = sphere_ibo;
        view.index_type    = VK_INDEX_TYPE_UINT32;
        view.use_ibo       = true;
        view.draw_count     = sphere_index_count;  // icosphere index count (uint32, small)
```
with:
```cpp
        view.ibo           = tmpl_ibo;
        view.index_type    = VK_INDEX_TYPE_UINT32;
        view.use_ibo       = true;
        view.draw_count     = tmpl_count;          // template index count (uint32, small)
```

- [ ] **Step 5: Size the cube to the cell (voxelHalfExtent) in the UBO**

In `lib/src/engine.cpp`, in the `marker_size` selection (currently `:3263`), replace:
```cpp
        float marker_size = view->desc.default_size;
        if (lod_context.active() && view->desc.type == ViewType::Markers
            && std::holds_alternative<MarkerOptions>(view->desc.options)
            && (std::get<MarkerOptions>(view->desc.options).render_mode
                   == MarkerOptions::RenderMode::Sphere3D
             || std::get<MarkerOptions>(view->desc.options).render_mode
                   == MarkerOptions::RenderMode::SphereMesh))
        {
            marker_size = lod_context.sphereRadius(view->desc.default_size);
        }
```
with:
```cpp
        float marker_size = view->desc.default_size;
        if (lod_context.active() && view->desc.type == ViewType::Markers
            && std::holds_alternative<MarkerOptions>(view->desc.options)
            && (std::get<MarkerOptions>(view->desc.options).render_mode
                   == MarkerOptions::RenderMode::Sphere3D
             || std::get<MarkerOptions>(view->desc.options).render_mode
                   == MarkerOptions::RenderMode::SphereMesh))
        {
            // Voxel LOD lit views (routed to SphereMesh) fill the cell: half-extent = 1/grid_n so the
            // cubes tile. Sphere LOD uses the cell-fill sphere radius scaled by --size.
            const bool voxel = options.lod_voxel && options.light_model != LightModel::None;
            marker_size = voxel ? lod_context.voxelHalfExtent()
                                : lod_context.sphereRadius(view->desc.default_size);
        }
```

- [ ] **Step 6: Build**

```bash
touch lib/src/remote.cpp
cmake --build build --target mimir -j 2>&1 | grep -iE "error|Built target mimir" | tail -3
cmake --build samples/remote-rendering/build -j 2>&1 | grep -iE "error|Built target rr-server" | tail -3
```
Expected: builds, no `error:`.

- [ ] **Step 7: Verify phong LOD renders cubes by default**

```bash
SCR=/tmp/uvl-t3; mkdir -p $SCR
SP=$SCR/phong-vox.log
nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 200 \
  --light-model phong --lod 64 --fps 60 > "$SP" 2>&1 &
PID=$!; for i in $(seq 1 30); do grep -q "listening on port" "$SP" && break; sleep 1; done
grep -iE "cube template|Raster LOD|forcing cell-center" "$SP" | head -3
timeout 40 samples/remote-rendering/build/rr-client 127.0.0.1 9000 "" tcp 40 2>&1 | tail -1
cp rr-client.ppm $SCR/phong-vox.ppm
python3 -c "from PIL import Image; Image.open('$SCR/phong-vox.ppm').save('$SCR/phong-vox.png')"
kill $PID 2>/dev/null
echo "inspect $SCR/phong-vox.png -- must show flat-shaded CUBES (blocky), not round spheres"
```
Expected: log shows `Voxel LOD: unit cube template built` + `forcing cell-center`; PNG shows blocky flat-shaded cubes. Open it to confirm (if only back-faces show / interior looks inside-out, the cube winding in Step 2 is reversed — swap each face's triangle order to `{base+0,base+2,base+1, base+0,base+3,base+2}` and rebuild).

- [ ] **Step 8: Verify phong-mesh voxels, PT voxels, and the sphere opt-out**

```bash
SCR=/tmp/uvl-t3
run() { # $1=extra args  $2=tag
  SP=$SCR/$2.log
  nohup samples/remote-rendering/build/rr-server 9000 640 480 20000000 413111 200 \
    $1 --lod 64 --fps 60 --spp 1 > "$SP" 2>&1 &
  local PID=$!; for i in $(seq 1 30); do grep -q "listening on port" "$SP" && break; sleep 1; done
  timeout 40 samples/remote-rendering/build/rr-client 127.0.0.1 9000 "" tcp 40 >/dev/null 2>&1
  cp rr-client.ppm $SCR/$2.ppm
  python3 -c "from PIL import Image; Image.open('$SCR/$2.ppm').save('$SCR/$2.png')"
  kill $PID 2>/dev/null
}
run "--light-model phong-mesh" phongmesh-vox   # cubes
run "--light-model path-tracing" pt-vox         # cubes (PT box branch)
run "--light-model phong --lod-shape sphere" phong-sph   # round spheres
run "--light-model none" none-vox               # flat points, no cubes
echo "inspect: phongmesh-vox/pt-vox = cubes; phong-sph = spheres; none-vox = points"
grep -L "forcing cell-center" $SCR/none-vox.log && echo "PASS: none did NOT force cell-center"
```
Expected: `phongmesh-vox.png` and `pt-vox.png` show cubes; `phong-sph.png` shows round spheres; `none-vox.png` shows flat points; `none` did not force cell-center.

- [ ] **Step 9: Perf sanity — phong voxel LOD is not slower than phong sphere LOD**

```bash
SCR=/tmp/uvl-t3
for shape in "" "--lod-shape sphere"; do
  SP=$SCR/perf$(echo $shape | tr -d ' -').log
  nohup samples/remote-rendering/build/rr-server 9000 640 480 100000000 413111 400 \
    --light-model phong --lod 128 $shape --fps 60 > "$SP" 2>&1 &
  PID=$!; for i in $(seq 1 40); do grep -q "listening on port" "$SP" && break; sleep 1; done
  timeout 60 samples/remote-rendering/build/rr-client 127.0.0.1 9000 "" tcp 200 >/dev/null 2>&1
  echo "shape='${shape:-voxel}':"; grep "ms render" "$SP" | tail -2
  kill $PID 2>/dev/null; sleep 1
done
```
Expected: voxel `render` ms is comparable to or lower than the sphere run (the draw is dominated by the LOD reduction either way; do not treat small noise as a regression).

- [ ] **Step 10: Commit**

```bash
git add lib/include/private/mimir/engine.hpp lib/src/engine.cpp
git commit -m "feat(rr): render lit LOD as instanced cube voxels (default)

Add a unit-cube template (ensureCubeMesh, per-face normals) and route
voxel-LOD lit views (phong included) through the SphereMesh instancing path
via a render_mode override at view creation, so phong/phong-mesh LOD render
as flat-shaded grid-aligned cubes by default. Cubes are sized to
voxelHalfExtent (cell-fill tiling); --lod-shape sphere reverts to spheres;
none is unaffected. PT already voxelized (Task 1).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CQfpwcA5mm3RDPqc27Kw3W"
```

---

## Self-Review

**Spec coverage:**
- Voxel default across lit models (phong/phong-mesh/PT) → Task 1 (PT), Task 3 (raster via render_mode override). ✓
- `none` excluded → Task 1 Step 3 condition, Task 3 Steps 3-5 conditions. ✓
- Remove `--lod-voxel`, add `--lod-shape sphere|voxel` → Task 1 Steps 4-7. ✓
- Full-cell tiling / voxelHalfExtent → Task 3 Step 5. ✓
- Cell-center forced (engine, all callers) → Task 1 Step 3. ✓
- Instanced cube mesh reusing SphereMesh path → Task 3 Steps 1-4. ✓
- Per-vertex normal (cube needs per-face) → Task 2. ✓
- Perf neutral-to-faster → Task 3 Step 9 sanity. ✓
- Non-goals (no cube impostor, no `none` voxels, no reduction change, keep phong-mesh) → respected; no task touches them. ✓

**Placeholder scan:** No TBD/TODO. Task 2 Step 4 contains an illustrative-then-corrected shader; the "use exactly this" corrected block is unambiguous. ✓

**Type consistency:** `lod_voxel` (options) set in Task 1, read in Task 1 Step 3, Task 3 Steps 3-5. `ensureCubeMesh`/`cube_vbo`/`cube_ibo`/`cube_index_count` declared Task 3 Step 1, defined/used Steps 2-4. Interleaved `{pos,normal}` layout defined Task 2 (Steps 1-4), consumed by the cube in Task 3 Step 2. `voxelHalfExtent()` already exists (prior feature). Names consistent. ✓
