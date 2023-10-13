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

[[vk::binding(3)]] Texture2D<half1> Tex2d_Half1;
[[vk::binding(3)]] Texture2D<half2> Tex2d_Half2;
[[vk::binding(3)]] Texture2D<half3> Tex2d_Half3;
[[vk::binding(3)]] Texture2D<half4> Tex2d_Half4;

[[vk::binding(3)]] Texture3D<half1> Tex3d_Half1;
[[vk::binding(3)]] Texture3D<half2> Tex3d_Half2;
[[vk::binding(3)]] Texture3D<half3> Tex3d_Half3;
[[vk::binding(3)]] Texture3D<half4> Tex3d_Half4;

[[vk::binding(3)]] Texture2D<int1> Tex2d_Int1;
[[vk::binding(3)]] Texture2D<int2> Tex2d_Int2;
[[vk::binding(3)]] Texture2D<int3> Tex2d_Int3;
[[vk::binding(3)]] Texture2D<int4> Tex2d_Int4;

[[vk::binding(3)]] Texture3D<int1> Tex3d_Int1;
[[vk::binding(3)]] Texture3D<int2> Tex3d_Int2;
[[vk::binding(3)]] Texture3D<int3> Tex3d_Int3;
[[vk::binding(3)]] Texture3D<int4> Tex3d_Int4;


[[vk::binding(4)]] SamplerState Sampler;


///////////////////////////////////////////////////////////////////////

[shader("fragment")]
float4 frag2d_Int1<U:IPicker>(VertexData input) : SV_Target
{
    float4 samp = float4(Tex2d_Int1.Sample(Sampler, input.uv).rrr, 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Int2<U:IPicker>(VertexData input) : SV_Target
{
    var samp = float4(Tex2d_Int2.Sample(Sampler, input.uv).rg, 0, 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Int3<U:IPicker>(VertexData input) : SV_Target
{
    var samp = float4(Tex2d_Int3.Sample(Sampler, input.uv), 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Int4<U:IPicker>(VertexData input) : SV_Target
{
    var samp = Tex2d_Int4.Sample(Sampler, input.uv);
    return U.getColor(samp);
}

///////////////////////////////////////////////////////////////////////

[shader("fragment")]
float4 frag2d_Float1<U:IPicker>(VertexData input) : SV_Target
{
    float4 samp = float4(Tex2d_Float1.Sample(Sampler, input.uv).rrr, 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Float2<U:IPicker>(VertexData input) : SV_Target
{
    var samp = float4(Tex2d_Float2.Sample(Sampler, input.uv).rg, 0, 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Float3<U:IPicker>(VertexData input) : SV_Target
{
    var samp = float4(Tex2d_Float3.Sample(Sampler, input.uv), 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Float4<U:IPicker>(VertexData input) : SV_Target
{
    var samp = Tex2d_Float4.Sample(Sampler, input.uv);
    return U.getColor(samp);
}

///////////////////////////////////////////////////////////////////////

[shader("fragment")]
float4 frag2d_Half1<U:IPicker>(VertexData input) : SV_Target
{
    float4 samp = float4(Tex2d_Half1.Sample(Sampler, input.uv).rrr, 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Half2<U:IPicker>(VertexData input) : SV_Target
{
    var samp = float4(Tex2d_Half2.Sample(Sampler, input.uv).rg, 0, 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Half3<U:IPicker>(VertexData input) : SV_Target
{
    var samp = float4(Tex2d_Half3.Sample(Sampler, input.uv), 1);
    return U.getColor(samp);
}

[shader("fragment")]
float4 frag2d_Half4<U:IPicker>(VertexData input) : SV_Target
{
    var samp = Tex2d_Half4.Sample(Sampler, input.uv);
    return U.getColor(samp);
}

///////////////////////////////////////////////////////////////////////

[shader("fragment")]
float4 frag3d_Int4(VertexData3d input) : SV_Target
{
    var samp = Tex3d_Int4.Sample(Sampler, input.uv);
    return float4(samp.rrr, 1.0);
}
