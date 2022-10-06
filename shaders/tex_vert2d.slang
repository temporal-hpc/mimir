import uniform;

[[vk::binding(0)]] cbuffer ModelViewProjectionUBO
{
  ModelViewProjection mvp;
};

struct VertexInput
{
  [[vk::location(0)]] float3 pos : POSITION;
  [[vk::location(1)]] float2 uv  : TEXCOORD;
};

struct VertexOutput
{
  float4 pos : SV_Position;
  float2 uv  : TEXCOORD;
};

VertexOutput main(VertexInput input)
{
  float4x4 view_proj = mul(mvp.view, mvp.proj);
  float4 pos = float4(input.pos, 1);
  pos.z = 0.99; // Temp hack to avoid z-fighting

  VertexOutput output;
  output.pos = mul(pos, view_proj);
  output.uv = input.uv;

  return output;
}
