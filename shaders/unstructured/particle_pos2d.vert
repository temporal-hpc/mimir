
struct ModelViewProjection
{
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

struct UniformDataParams
{
  int3 extent;
};

cbuffer ModelViewProjectionUBO : register(b0)
{
  ModelViewProjection mvp;
};

cbuffer UniformDataParamsUBO : register(b1)
{
  UniformDataParams params;
};

struct VertexInput
{
  [[vk::location(0)]] float2 pos : POSITION0;
};

struct VertexOutput
{
  [[vk::builtin("PointSize")]] float point_size;
  float4 pos : SV_Position;
  float2 center_pos : POSITION1;
  float4 color : COLOR;
  float marker_size;
  //float line_width;
};

VertexOutput main(VertexInput input)
{
  //float4x4 mat = transpose(mvp.proj * mvp.view * mvp.model);

  VertexOutput output;
  output.point_size = 10.f;
  float4 pos = float4(2.f * (input.pos / params.extent.xy) - 1.f, 0.f, 1.f);
  output.pos = mul(pos, mvp.model);
  output.center_pos = ((output.pos.xy / output.pos.w) + 1.f) * 0.5 * float2(800, 600);
  output.color = float4(0, 0, 1, 1);
  output.marker_size = output.point_size;
  //output.line_width = 1;

  return output;
}
