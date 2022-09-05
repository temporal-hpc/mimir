[[vk::binding(5)]] Texture3D<int4> Texture;
[[vk::binding(6)]] SamplerState Sampler;

struct FragmentInput
{
  float4 pos : SV_Position;
  float3 uv  : TEXCOORD;
};

float4 main(FragmentInput input) : SV_Target
{
  int4 color = Texture.Sample(Sampler, input.uv);
  return float4(color.rrr, 1.0);
}
