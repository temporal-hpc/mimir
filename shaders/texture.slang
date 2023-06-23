import uniforms;

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> scene;
ConstantBuffer<PrimitiveParams> primitive;

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
    pos.z = pos.z - primitive.depth;

    VertexData3d output;
    output.pos = mul(pos, view_proj);
    output.uv  = float3(input.uv, primitive.depth);

    return output;
}

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

[[vk::binding(3)]] Texture1D<float3> Tex1d32x3;
[[vk::binding(3)]] Texture2D<float3> Tex2d32x3;
[[vk::binding(3)]] Texture3D<float3> Tex3d32x3;

[[vk::binding(3)]] Texture2D<float3> TextureFlt;
[[vk::binding(3)]] Texture3D<int4> TextureRGBA;
SamplerState Sampler;

interface ITexture
{
    float4 getColor(float2 uv);
}

struct Texture2dFloat3 : ITexture
{
    float4 getColor(float2 uv)
    {
        return float4(Tex2d32x3.Sample(Sampler, uv), 1);
    }
}

[shader("fragment")]
float4 fragment2dMain<U:IColorable>(VertexData input) : SV_Target
{
    Texture2dFloat3 tex;
    var sample = tex.getColor(input.uv);
    U colorable;
    var color = colorable.getColor(sample.xyz);
    return float4(color, 1);
}

[shader("fragment")]
float4 fragment3dMain(VertexData3d input) : SV_Target
{
    var color = TextureRGBA.Sample(Sampler, input.uv);
    return float4(color.rrr, 1.0);
}
