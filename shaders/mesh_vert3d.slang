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
  [[vk::location(0)]] float3 pos : POSITION;
};

struct VertexOutput
{
  float4 pos        : SV_Position;
  //float4 color      : COLOR;
  //float marker_size;
};

VertexOutput main(VertexInput input)
{
  float4x4 view_proj = mul(mvp.proj, mvp.view);

  VertexOutput output;
  float4 pos = float4(2.0 * (input.pos / params.extent) - 1.0, 1.0);
  output.pos = mul(view_proj, pos);
  //output.color = float4(0, 0, 1, 1);
  //output.marker_size = 10.f;

  return output;
}
