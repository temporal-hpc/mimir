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
  [[vk::location(1)]] float2 uv  : TEXCOORD;
};

struct VertexOutput
{
  float4 pos : SV_Position;
  float2 uv  : TEXCOORD;
};

VertexOutput main(VertexInput input)
{
  float4 pos = float4(input.pos, 1.0);

  VertexOutput output;
  output.pos = mul(pos, mvp.model);
  output.uv = input.uv;

  return output;
}
