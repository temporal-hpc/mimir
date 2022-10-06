import uniform;

[[vk::binding(0)]] cbuffer ModelViewProjectionUBO
{
  ModelViewProjection mvp;
};

[[vk::binding(1)]] cbuffer UniformDataParamsUBO
{
  UniformDataParams params;
};

struct VertexInput
{
  [[vk::location(0)]] float2 pos : POSITION;
};

struct VertexOutput
{
  float4 pos    : SV_Position;
  float4 center : POSITION;
  float4 color  : COLOR;
  //float marker_size;
};

VertexOutput main(VertexInput input)
{
  float4x4 view_proj = mul(mvp.view, mvp.proj);

  VertexOutput output;
  output.center = float4(2 * (float3(input.pos, 1) / params.extent) - 1, 1);
  output.pos    = mul(output.center, view_proj);
  output.color  = float4(0, 0, 1, 1);
  //output.marker_size = 10.f;

  return output;
}
