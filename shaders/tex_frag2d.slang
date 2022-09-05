[[vk::binding(5)]] Texture2D<float> Texture;
[[vk::binding(6)]] SamplerState Sampler;

struct FragmentInput
{
  float4 pos : SV_Position;
  float2 uv  : TEXCOORD;
};

float4 main(FragmentInput input) : SV_Target
{
  float dist = Texture.Sample(Sampler, input.uv);
  return float4(float3(dist), 1);
}
