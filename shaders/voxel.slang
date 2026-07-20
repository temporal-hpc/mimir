import uniforms;

import position; // IPosition
import color;    // IColor
import size;     // ISize

struct VertexData
{
    float4 center : POSITION;
    float4 color  : COLOR;
};

[shader("vertex")]
VertexData vertexMain<P : IPosition, C : IColor, S : ISize>(
    P in_pos: POSITION, C in_color: COLOR, S in_size: PSIZE)
{
    VertexData output;
    output.center = in_pos.getPosition();
    output.color = in_color.getColor();

    return output;
}

struct VoxelData
{
    float4 pos    : SV_Position;
    float2 uv     : TEXCOORD;
    float4 color  : COLOR;
    float3 normal : NORMAL;
};

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain2D(point VertexData input[1], inout TriangleStream<VoxelData> stream)
{
    float3 center = input[0].center.xyz;

    // Compute offsets for block (voxel) vertices
    float3 sz = float3(view.default_size);
    float3 offsets[3] = {
        float3(sz.x, 0, 0),
        float3(0, sz.y, 0),
        float3(0, 0, sz.z)
    };

    VoxelData output;
    output.color = input[0].color;
    float4 unit = float4(0);
    int k = 2;
    unit[k] = 1;
    // float4 normal = mul(unit, model_view);
    // output.normal = normal.xyz;
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

[shader("geometry")]
[maxvertexcount(24)]
void geometryMain3D(point VertexData input[1], inout TriangleStream<VoxelData> stream)
{
    float3 center = input[0].center.xyz;

    // Compute offsets for block (voxel) vertices
    float3 sz = float3(view.default_size);
    float3 offsets[3] = {
        float3(sz.x, 0, 0),
        float3(0, sz.y, 0),
        float3(0, 0, sz.z)
    };

    VoxelData output;
    output.color = input[0].color;
    float2 texcoords[4] = { {0,0}, {0,1}, {1,0}, {1,1} };
    // Emit ALL 6 faces of the cube. The old code drew only 3 faces, culled by dot(-center, normal) > 0,
    // which assumes the viewer is at the world origin -- wrong once the grid is not centered on the eye,
    // so camera-facing faces got culled and cubes looked hollow/misaligned. Cull mode is NONE and depth
    // testing resolves occlusion for opaque voxels; translucent voxels correctly blend front+back faces.
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
            unit[k] = sgn;                 // outward normal for this face (used by phong shading)
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

float3 lambert(float3 normal, float3 light, float3 color)
{
    return color * max(dot(normalize(normal), normalize(light)), 0);
}

// Shade a cube face by its (axis-aligned) normal under the scene light. Kept simple: Lambert diffuse
// plus a fixed ambient term so back faces stay visible. --light-mode phong sets view.shading = 1.
float4 shadeVoxel(float4 color, float3 normal)
{
    if (view.shading < 0.5) { return color; }
    float3 N = normalize(normal);
    float3 L = normalize(scene.light_pos); // direction TO the light (matches the path tracer's sun)
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
