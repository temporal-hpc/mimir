import uniform;

[[vk::binding(0)]] cbuffer ModelViewProjectionUBO
{
  ModelViewProjection mvp;
};

[[vk::binding(1)]] cbuffer SceneParamsUBO
{
  SceneParams scene;
};

struct VertexInput
{
  [[vk::location(0)]] float3 pos : POSITION;
  [[vk::location(1)]] float2 uv  : TEXCOORD;
};

struct VertexOutput
{
  float4 pos : SV_Position;
  float3 uv  : TEXCOORD;
};

VertexOutput main(VertexInput input)
{
  float4x4 view_proj = mul(mvp.view, mvp.proj);
  float4 pos = float4(input.pos, 1);
  pos.z = pos.z - scene.depth;

  VertexOutput output;
  output.pos = mul(pos, view_proj);
  output.uv  = float3(input.uv, scene.depth);

  return output;
}
