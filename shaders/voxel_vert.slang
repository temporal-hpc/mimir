struct ModelViewProjection
{
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

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
  float3 center : POSITION;
  float4 color  : COLOR;
};

VertexOutput main(VertexInput input)
{
  VertexOutput output;
  output.center = 2.0 * (input.pos / float3(200.0)) - 1.0;
  //output.pos = mul(float4(output.center, 1.0), mvp.model);
  output.color = float4(.5, .5, .5, 1);

  return output;
}
