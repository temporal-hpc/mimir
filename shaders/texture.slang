import uniform;

[[vk::binding(0)]] ConstantBuffer<ModelViewProjection> mvp;
[[vk::binding(1)]] ConstantBuffer<SceneParams> scene;

struct VertexInput
{
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] float2 uv  : TEXCOORD;
};

struct VertexData
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD;
};

[shader("vertex")]
VertexData vertex2dMain(VertexInput input)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);
    float4 pos = float4(input.pos, 1);
    pos.z = 0.99; // Temp hack to avoid z-fighting

    VertexData output;
    output.pos = mul(pos, view_proj);
    output.uv = input.uv;

    return output;
}

struct VertexData3d
{
    float4 pos : SV_Position;
    float3 uv  : TEXCOORD;
};

[shader("vertex")]
VertexData3d vertex3dMain(VertexInput input)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);
    float4 pos = float4(input.pos, 1);
    pos.z = pos.z - scene.depth;

    VertexData3d output;
    output.pos = mul(pos, view_proj);
    output.uv  = float3(input.uv, scene.depth);

    return output;
}

[[vk::binding(5)]] Texture2D<float3> TextureFlt;
[[vk::binding(6)]] SamplerState Sampler;

interface IColorable
{
    float3 getColor(float3 rgb);
}

struct RawColor : IColorable
{
    float3 getColor(float3 rgb)
    {
        return rgb;
    }
}

struct Grayscale : IColorable
{
    float3 getColor(float3 rgb)
    {
        return float3(0.299*rgb.r + 0.587*rgb.g + 0.114*rgb.b);
    }
}

[shader("fragment")]
float4 fragment2dMain<T : IColorable>(VertexData input) : SV_Target
{
    var sample = TextureFlt.Sample(Sampler, input.uv);
    T colorable;
    var color = colorable.getColor(sample);
    return float4(color, 1);
}

[[vk::binding(5)]] Texture3D<int4> TextureRGBA;

[shader("fragment")]
float4 fragment3dMain(VertexData3d input) : SV_Target
{
    var color = TextureRGBA.Sample(Sampler, input.uv);
    return float4(color.rrr, 1.0);
}
