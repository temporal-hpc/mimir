import uniforms;

import position; // IPosition
import color;    // IColor
import size;     // ISize

struct VertexData
{
    float4 center : POSITION;
    float4 color  : COLOR;
};

[[vk::binding(5)]] StructuredBuffer<float4> buffer;

[shader("vertex")]
VertexData vertexMainIndirect(PositionFloat3 in_pos, int index)
{
    VertexData output;
    output.center = in_pos.getPosition(scene.extent);
    output.color = buffer[index];

    return output;
}

[shader("vertex")]
VertexData vertexMain<P : IPosition, C : IColor, S : ISize>(
    P in_pos: POSITION, C in_color: COLOR, S in_size: PSIZE)
{
    VertexData output;
    output.center = in_pos.getPosition(scene.extent);
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
    float4x4 model_view = mul(mvp.model, mvp.view);
    float4 center = mul(input[0].center, model_view);
    // In orthographic projection we have to fix our origin (center),
    // because every ray has the same direction
    if (mvp.proj[3][3] == 1)
    {
        center = float4(0, 0, -1, 1);
    }

    // Compute offsets for block (voxel) vertices
    float3 vox_size = float3(view.default_size) / scene.extent;
    float4 offsets[3] = {
        model_view[0] * vox_size.x,
        model_view[1] * vox_size.y,
        model_view[2] * vox_size.z
    };

    VoxelData output;
    output.color = input[0].color;
    float4 unit = float4(0);
    int k = 2;
    unit[k] = 1;
    float4 normal = mul(unit, model_view);
    output.normal = normal.xyz;
    float4 face_offset;
    float4 vertex_offsets[4];
    float4 dx = offsets[(k + 2) % 3];
    float4 dy = offsets[(k + 1) % 3];
    float2 texcoords[4];
    if (dot(-center.xyz, normal.xyz) > 0)
    {
        texcoords = {{0,0}, {0,1}, {1,0}, {1,1}};
        vertex_offsets = {(-dx-dy), (-dx+dy), (+dx-dy), (+dx+dy)};
    }
    else
    {
        texcoords = {{0,0}, {1,0}, {0,1}, {1,1}};
        vertex_offsets = {(-dx-dy), (+dx-dy), (-dx+dy), (+dx+dy)};
    }
    for (int i = 0; i < 4; ++i)
    {
        float4 v = center + vertex_offsets[i];
        output.pos = mul(v, mvp.proj);
        output.uv = texcoords[i];
        stream.Append(output);
    }
}

[shader("geometry")]
[maxvertexcount(12)]
void geometryMain3D(point VertexData input[1], inout TriangleStream<VoxelData> stream)
{
    float4x4 model_view = mul(mvp.model, mvp.view);
    float4 center = mul(input[0].center, model_view);
    // In orthographic projection we have to fix our origin (center),
    // because every ray has the same direction
    if (mvp.proj[3][3] == 1)
    {
        center = float4(0, 0, -1, 1);
    }

    // Compute offsets for block (voxel) vertices
    float3 vox_size = .001 * float3(view.default_size);
    float4 offsets[3] = {
        model_view[0] * vox_size.x,
        model_view[1] * vox_size.y,
        model_view[2] * vox_size.z
    };

    VoxelData output;
    output.color = input[0].color;
    for (int k = 0; k < 3; ++k)
    {
        float4 unit = float4(0);
        unit[k] = 1;
        float4 normal = mul(unit, model_view);
        output.normal = normal.xyz;
        float4 face_offset;
        float4 vertex_offsets[4];
        float4 dx = offsets[(k + 2) % 3];
        float4 dy = offsets[(k + 1) % 3];
        float2 texcoords[4];
        if (dot(-center.xyz, normal.xyz) > 0)
        {
            face_offset = offsets[k];
            texcoords = {{0,0}, {0,1}, {1,0}, {1,1}};
            vertex_offsets = {(-dx-dy), (-dx+dy), (+dx-dy), (+dx+dy)};
        }
        else
        {
            face_offset = -offsets[k];
            texcoords = {{0,0}, {1,0}, {0,1}, {1,1}};
            vertex_offsets = {(-dx-dy), (+dx-dy), (-dx+dy), (+dx+dy)};
        }
        for (int i = 0; i < 4; ++i)
        {
            float4 v = center + face_offset + vertex_offsets[i];
            output.pos = mul(v, mvp.proj);
            output.uv = texcoords[i];
            stream.Append(output);
        }
        stream.RestartStrip();
    }
}

float3 lambert(float3 normal, float3 light, float3 color)
{
    return color * max(dot(normalize(normal), normalize(light)), 0);
}

[shader("fragment")]
float4 fragmentMain(VoxelData input) : SV_Target
{
    return input.color;
    // float2 p = 2 * input.uv - 1;
    // float distance = disc(p * 10, view.size);
    // return filled(distance, 1, 1, view.color, scene.bg_color);
}
