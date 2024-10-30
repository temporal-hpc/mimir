import uniforms;

struct VertexInput
{
    float3 pos : POSITION;
    [[vk::location(5)]] float2 uv  : TEXCOORD;
};

struct VertexData
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD;
};

[shader("vertex")]
VertexData vertexMain(VertexInput input)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);
    float4 pos = float4(input.pos, 1);

    VertexData output;
    output.pos = mul(pos, view_proj);
    output.uv = input.uv;

    return output;
}

interface IPicker
{
    static float4 getColor(float4 rgba);
}

struct RawColor : IPicker
{
    static float4 getColor(float4 rgba)
    {
        return rgba;
    }
}

struct Grayscale : IPicker
{
    static float4 getColor(float4 rgba)
    {
        float x = 0.299*rgba.r + 0.587*rgba.g + 0.114*rgba.b;
        return float4(x, x, x, rgba.a);
    }
}

[[vk::binding(3)]] Texture2D<float1> Tex2d_Float1;
[[vk::binding(3)]] Texture2D<float2> Tex2d_Float2;
[[vk::binding(3)]] Texture2D<float3> Tex2d_Float3;
[[vk::binding(3)]] Texture2D<float4> Tex2d_Float4;

[[vk::binding(3)]] Texture3D<float1> Tex3d_Float1;
[[vk::binding(3)]] Texture3D<float2> Tex3d_Float2;
[[vk::binding(3)]] Texture3D<float3> Tex3d_Float3;
[[vk::binding(3)]] Texture3D<float4> Tex3d_Float4;

[[vk::binding(3)]] Texture3D<int> Tex3d_Int1;

[[vk::binding(4)]] SamplerState Sampler;

[shader("fragment")]
float4 fragmentMain(VertexData input) : SV_Target
{
    IPicker picker = RawColor();
    var samp = Tex2d_Float4.Sample(Sampler, input.uv);
    return picker.getColor(samp);
}
