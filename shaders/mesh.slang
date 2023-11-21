import uniforms;

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> params;
ConstantBuffer<ViewParams> view;

struct VertexInput
{
    [[vk::location(0)]] float2 pos : POSITION;
};

struct VertexData
{
    float4 pos   : SV_Position;
    float4 color : COLOR;
    //float marker_size;
};

[shader("vertex")]
VertexData vertex2dMain(VertexInput input)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);

    VertexData output;
    float2 extent = params.extent.xy;
    float4 pos = float4(2 * (input.pos / extent) - 1, 0, 1);
    output.pos = mul(pos, view_proj);
    output.color = view.default_color;
    //output.marker_size = 10.f;

    return output;
}

struct VertexInput3d
{
    [[vk::location(0)]] float3 pos : POSITION;
};

[shader("vertex")]
VertexData vertex3dMain(VertexInput3d input)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);

    VertexData output;
    float4 pos = float4(2 * (input.pos / params.extent) - 1, 1);
    output.pos = mul(pos, view_proj);
    output.color = view.default_color;
    //output.marker_size = 10.f;

    return output;
}

[shader("fragment")]
float4 fragmentMain(VertexData input) : SV_Target
{
    return input.color;
}
