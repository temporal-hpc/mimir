import uniform;

[[vk::binding(0)]] cbuffer ModelViewProjectionUBO
{
  ModelViewProjection mvp;
};

struct VertexInput
{
  [[vk::location(0)]] float3 pos : POSITION;
  [[vk::location(1)]] int color  : COLOR;
};

struct VertexOutput
{
  float4 pos    : SV_Position;
  float4 center : POSITION;
  float4 color  : COLOR;
};

VertexOutput main(VertexInput input)
{
  float4x4 view_proj = mul(mvp.view, mvp.proj);

  VertexOutput output;
  output.center = float4(input.pos, 1);
  output.pos    = mul(output.center, view_proj);
  output.color  = float4(float1(input.color).xxx, 1);

  return output;
}
