import uniform;

[[vk::binding(0)]] cbuffer ModelViewProjectionUBO
{
  ModelViewProjection mvp;
};

struct VertexInput
{
  [[vk::location(0)]] float3 pos : POSITION;
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
  output.center = float4(2 * (input.pos / float3(200)) - 1, 1);
  output.pos = mul(output.center, view_proj);
  output.color = float4(.5, .5, .5, 1);

  return output;
}
