// In-shader grid-coarsening LOD for ViewType::Voxels.
//
// Draws M^3 procedural points (vkCmdDraw(M^3,1,0,0), NO vertex buffers): the vertex stage derives each
// coarse cell (cx,cy,cz) from SV_VertexID, max-pools its disjoint [c*N/M,(c+1)*N/M) block of the FINE
// N^3 int-state grid (bound as an SSBO), looks up colorbuf[max], and emits the coarse cube's world
// center + per-axis block dimension. The geometry stage expands the point to a cube exactly like
// voxel.slang, but sizes it by the block dimension so coarse cubes tile the same world box as the fine
// grid (and reduce to the fine voxels when block == 1, i.e. M == N).
//
// SELF-CONTAINED by design: lib/src/shader.cpp loads ONE module per pipeline and resolves every
// entrypoint from it (findEntryPointByName), so a pipeline's stages cannot span modules. geometryMain2D/3D
// and fragmentMain below are COPIES of voxel.slang's (the only change is `sz` reading input[0].blockdim
// instead of a scalar view.default_size). Keep them in sync with voxel.slang if that shader changes.

import uniforms; // mvp, scene, view, colorbuf (binding 5 = colormap)

// MUST stay layout-compatible between the vertex and geometry stages below. `blockdim` is new relative
// to voxel.slang's VertexData: the coarse cell's fine-block size per axis (1 for the fine grid).
struct VertexData
{
    float4 center   : POSITION;
    float4 color    : COLOR;
    float4 blockdim : TEXCOORD0;
};

struct LodPush
{
    uint   fineN;       // fine grid resolution per axis (N)
    uint   coarseM;     // coarse grid resolution per axis (M)
    float4 gridOrigin;  // fine grid world origin (xyz); mirrors makeStructuredGrid's `start`
    float4 gridSpacing; // fine grid world spacing per cell (xyz); makeStructuredGrid uses 1
};
[vk::push_constant] LodPush lod;

// Fine N^3 states, row-major x + N*(y + N*z). Set 1 so set 0 stays the shared uniforms layout.
[vk::binding(0, 1)] StructuredBuffer<int> fineState;

[shader("vertex")]
VertexData vertexLodMain(uint vid : SV_VertexID)
{
    uint N = lod.fineN, M = lod.coarseM;
    uint cx = vid % M, cy = (vid / M) % M, cz = vid / (M * M);

    // Disjoint fine block [c*N/M, (c+1)*N/M) per axis (integer floor handles non-divisible N/M).
    uint x0 = cx * N / M, x1 = (cx + 1) * N / M;
    uint y0 = cy * N / M, y1 = (cy + 1) * N / M;
    uint z0 = cz * N / M, z1 = (cz + 1) * N / M;

    // Max-pool the block: state 0 = dead, larger = "most-alive present".
    int m = 0;
    for (uint z = z0; z < z1; ++z)
      for (uint y = y0; y < y1; ++y)
        for (uint x = x0; x < x1; ++x)
            m = max(m, fineState[x + N * (y + N * z)]);

    // Coarse cube center = center-of-mass of the block's fine-cube extents. Fine cell i has its cube
    // centered at origin + spacing*i, so the block [lo,hi) spans cell centers origin+spacing*lo ..
    // origin+spacing*(hi-1); its midpoint is origin + spacing*(lo + hi - 1)/2. blockdim = hi - lo (the
    // fine-cell count per axis) scales the cube in the geometry stage. Both reduce to the fine voxel
    // (center origin+spacing*i, blockdim 1) when M == N.
    float3 lo3 = float3(x0, y0, z0);
    float3 hi3 = float3(x1, y1, z1);
    float3 center = lod.gridOrigin.xyz + lod.gridSpacing.xyz * (lo3 + hi3 - 1.0) * 0.5;

    VertexData o;
    o.center   = float4(center, 1.0);
    o.color    = colorbuf[m];
    o.blockdim = float4(hi3 - lo3, 0.0);
    return o;
}

struct VoxelData
{
    float4 pos    : SV_Position;
    float2 uv     : TEXCOORD;
    float4 color  : COLOR;
    float3 normal : NORMAL;
};

// --- Copied from voxel.slang (geometryMain2D), sz scaled by the per-cell block dimension. ---
[shader("geometry")]
[maxvertexcount(4)]
void geometryMain2D(point VertexData input[1], inout TriangleStream<VoxelData> stream)
{
    float3 center = input[0].center.xyz;

    float3 sz = float3(view.default_size) * input[0].blockdim.xyz;
    float3 offsets[3] = {
        float3(sz.x, 0, 0),
        float3(0, sz.y, 0),
        float3(0, 0, sz.z)
    };

    VoxelData output;
    output.color = input[0].color;
    int k = 2;
    float3 dx = offsets[(k + 2) % 3];
    float3 dy = offsets[(k + 1) % 3];
    float3 vertices[4]  = {(-dx-dy), (-dx+dy), (+dx-dy), (+dx+dy)};
    float2 texcoords[4] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    for (int i = 0; i < 4; ++i)
    {
        float3 v = center + vertices[i];
        output.pos = mul(float4(v, 1.0), mvp.all);
        output.uv = texcoords[i];
        stream.Append(output);
    }
}

// --- Copied from voxel.slang (geometryMain3D), sz scaled by the per-cell block dimension. All 6
//     faces are emitted (see the note in voxel.slang); cull mode NONE + depth test resolve them. ---
[shader("geometry")]
[maxvertexcount(24)]
void geometryMain3D(point VertexData input[1], inout TriangleStream<VoxelData> stream)
{
    float3 center = input[0].center.xyz;

    float3 sz = float3(view.default_size) * input[0].blockdim.xyz;
    float3 offsets[3] = {
        float3(sz.x, 0, 0),
        float3(0, sz.y, 0),
        float3(0, 0, sz.z)
    };

    VoxelData output;
    output.color = input[0].color;
    float2 texcoords[4] = { {0,0}, {0,1}, {1,0}, {1,1} };
    for (int k = 0; k < 3; ++k)
    {
        float3 dx = offsets[(k + 2) % 3];
        float3 dy = offsets[(k + 1) % 3];
        float3 vertex_offsets[4] = { (-dx-dy), (-dx+dy), (+dx-dy), (+dx+dy) };
        [unroll]
        for (int s = 0; s < 2; ++s)
        {
            float sgn = (s == 0) ? 1.0 : -1.0;
            float3 unit = float3(0);
            unit[k] = sgn;
            output.normal = unit;
            float3 face_offset = sgn * offsets[k];
            for (int i = 0; i < 4; ++i)
            {
                float3 v = center + face_offset + vertex_offsets[i];
                output.pos = mul(float4(v, 1.0), mvp.all);
                output.uv = texcoords[i];
                stream.Append(output);
            }
            stream.RestartStrip();
        }
    }
}

// --- Copied from voxel.slang (shadeVoxel + fragmentMain). Keep in sync with voxel.slang. ---
float4 shadeVoxel(float4 color, float3 normal)
{
    if (view.shading < 0.5) { return color; }
    float3 N = normalize(normal);
    float3 L = normalize(-scene.light_pos);
    float  diffuse = max(dot(N, L), 0.0);
    const float ambient = 0.4;
    float3 lit = color.rgb * (ambient + (1.0 - ambient) * diffuse);
    return float4(saturate(lit), color.a);
}

[shader("fragment")]
float4 fragmentMain(VoxelData input) : SV_Target
{
    return shadeVoxel(input.color, input.normal);
}
