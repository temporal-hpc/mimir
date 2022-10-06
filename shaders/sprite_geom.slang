import uniform;

[[vk::binding(0)]] cbuffer ModelViewProjectionUBO
{
  ModelViewProjection mvp;
};

struct GeometryInput
{
  float4 pos    : SV_Position;
  float4 center : POSITION;
  float4 color  : COLOR;
};

struct GeometryOutput
{
  float4 pos   : SV_Position;
  float4 color : COLOR;
  float2 uv    : TEXCOORD;
  //float3 normal : NORMAL;
};

[maxvertexcount(4)]
void main(point GeometryInput input[1], inout TriangleStream<GeometryOutput> stream)
{
  float4x4 model_view = mul(mvp.model, mvp.view);
  float4 center = mul(input[0].center, model_view);
  float3 half_block = .5 * float3(.06);

  float4 dx = model_view[0] * half_block.x;
  float4 dy = model_view[1] * half_block.y;
  float3 n = float3(0, 0, -1);

  // In orthographic projection we have to fix our origin (center),
  // because every ray has the same direction
  if (mvp.proj[3][3] == 1)
  {
    center = float4(0, 0, -1, 1);
  }

  // Emit a primitive only if the sign of the dot product is positive
  float4 normal = mul(float4(n, 0), model_view);

  GeometryOutput output;
  output.color = input[0].color;
  //output.normal = normal.xyz;

  // Add point coordinates
  float4 v1 = center + (-dx - dy);
  output.pos = mul(v1, mvp.proj);
  output.uv = float2(0, 0);
  stream.Append(output);

  float4 v2 = center + (-dx + dy);
  output.pos = mul(v2, mvp.proj);
  output.uv = float2(0, 1);
  stream.Append(output);

  float4 v3 = center + (+dx - dy);
  output.pos = mul(v3, mvp.proj);
  output.uv = float2(1, 0);
  stream.Append(output);

  float4 v4 = center + (+dx + dy);
  output.pos = mul(v4, mvp.proj);
  output.uv = float2(1, 1);
  stream.Append(output);
}
