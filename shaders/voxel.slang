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
[maxvertexcount(12)]
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
    for (int k = 0; k < 3; ++k)
    {
        float3 unit = float3(0);
        unit[k] = 1;
        output.normal = unit;
        float3 face_offset;
        float3 vertex_offsets[4];
        float3 dx = offsets[(k + 2) % 3];
        float3 dy = offsets[(k + 1) % 3];
        float2 texcoords[4];
        if (dot(-center.xyz, output.normal) > 0)
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
            float3 v = center + face_offset + vertex_offsets[i];
            output.pos = mul(float4(v, 1.0), mvp.all);
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
    // return filled(distance, 1, 1, view.color, scene.background_color);
}
