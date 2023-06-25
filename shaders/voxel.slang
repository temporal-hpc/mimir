import uniforms;

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> scene;
ConstantBuffer<PrimitiveParams> primitive;

struct VertexData
{
    float4 center : POSITION;
    float4 color  : COLOR;
};

struct VertexInput
{
    [[vk::location(0)]] float3 pos : POSITION;
};

[shader("vertex")]
VertexData vertexMain(VertexInput input)
{
    VertexData output;
    output.center = float4(2 * (input.pos / scene.extent) - 1, 1);
    output.color = primitive.color;

    return output;
}

struct VertexInputImplicit
{
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] int color  : COLOR;
};

[shader("vertex")]
VertexData vertexImplicitMain(VertexInputImplicit input)
{
    VertexData output;
    output.center = float4(input.pos, 1);
    output.color  = float4(float1(input.color).xxx, 1);

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
[maxvertexcount(12)]
void geometryMain(point VertexData input[1], inout TriangleStream<VoxelData> stream)
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
    float3 vox_size = .001 * float3(primitive.size);
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

float4 filled(float distance, float linewidth, float antialias, float4 fill, float4 stroke)
{
    float t = linewidth / 2.0 - antialias;
    float signed_distance = distance;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance / antialias;
    alpha = exp(-alpha * alpha);
    if (border_distance < 0.0)      return fill;
    else if (signed_distance < 0.0) return fill;
    else                            return float4(fill.rgb, alpha * fill.a);
}
float disc(float2 p, float size)
{
    return length(p) - size/2;
}

[shader("fragment")]
float4 fragmentMain(VoxelData input) : SV_Target
{
    float3 light_front = float3(0, 0, 1);
    float3 light_up = float3(0, 1, 0);
    float3 light_color = input.color.xyz;

    float front_bias = .9;
    float3 ambient_light = float3(.1);
    float3 color_front = front_bias * lambert(input.normal, light_front, light_color);
    float3 color_up = (1 - front_bias) * lambert(input.normal, light_up, light_color);

    return float4(ambient_light + color_front + color_up, input.color.w);
    // float2 p = 2 * input.uv - 1;
    // float distance = disc(p * 10, primitive.size);
    // return filled(distance, 1, 1, primitive.color, scene.bg_color);
}
