import uniform;

[[vk::binding(0)]] ConstantBuffer<ModelViewProjection> mvp;
[[vk::binding(1)]] ConstantBuffer<SceneParams> scene;
[[vk::binding(2)]] ConstantBuffer<PrimitiveParams> primitive;

struct VertexInput
{
    [[vk::location(0)]] float3 pos : POSITION;
};

struct VertexData
{
    float4 pos    : SV_Position;
    float4 center : POSITION;
    float4 color  : COLOR;
};

[shader("vertex")]
VertexData vertexMain(VertexInput input)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);

    VertexData output;
    output.center = float4(2 * (input.pos / float3(200)) - 1, 1);
    output.pos = mul(output.center, view_proj);
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
    float4x4 view_proj = mul(mvp.view, mvp.proj);

    VertexData output;
    output.center = float4(input.pos, 1);
    output.pos    = mul(output.center, view_proj);
    output.color  = float4(float1(input.color).xxx, 1);

    return output;
}

struct VoxelData
{
    float4 pos    : SV_Position;
    float4 color  : COLOR;
    float3 normal : NORMAL;
};

[shader("geometry")]
[maxvertexcount(12)]
void geometryMain(point VertexData input[1], inout TriangleStream<VoxelData> stream)
{
    float4 pos = input[0].pos;

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
        float4 shift, dx, dy;
        if (dot(-center.xyz, normal.xyz) > 0)
        {
            shift = offsets[k];
            dx = offsets[(k + 2) % 3];
            dy = offsets[(k + 1) % 3];
        }
        else
        {
            shift = -offsets[k];
            dx = offsets[(k + 1) % 3];
            dy = offsets[(k + 2) % 3];
        }
        float4 corners[4] = {(+dx-dy), (-dx-dy), (+dx+dy), (-dx+dy)};
        for (int i = 0; i < 4; ++i)
        {
            float4 v = center + shift + corners[i];
            output.pos = mul(v, mvp.proj);
            stream.Append(output);
        }
    }
}

float3 lambert(float3 normal, float3 light, float3 color)
{
    return color * max(dot(normalize(normal), normalize(light)), 0);
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
}
