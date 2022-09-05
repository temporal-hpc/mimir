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

struct GeometryInput
{
  float4 pos   : SV_Position;
  float4 color : COLOR;
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
  float4x4 model_view = mvp.view * mvp.model;
  float4 center = mul(model_view, input[0].pos);
  float3 half_block = .5 * float3(1.0, 1.0, 1.0);

  float4 dx = model_view[0] * half_block.x;
  float4 dy = model_view[1] * half_block.y;
  float4 shift = -model_view[2] * half_block.z;
  float3 n = float3(0.0, 0.0, -1.0);

  float4 v1 = (center + shift) + (dx - dy);
  float4 v2 = (center + shift) + (-dx - dy);
  float4 v3 = (center + shift) + (dx + dy);
  float4 v4 = (center + shift) + (-dx + dy);

  // In orthographic projection we have to fix our origin (center),
  // because every ray has the same direction
  if (mvp.proj[3][3] == 1.f)
  {
    center = float4(0.0, 0.0, -1.0, 1.0);
  }

  // Emit a primitive only if the sign of the dot product is positive
  float4 normal = mul(mvp.view * mvp.model, float4(n, 0.0));

  GeometryOutput output;
  output.color = input[0].color;
  //output.normal = normal.xyz;

  // Add point coordinates
  float4 pos = input[0].pos;

  output.pos = pos + float4(-.03, -.03, 0, 0); //mvp.proj * v1;
  output.uv = float2(0.f, 0.f);
  stream.Append(output);

  output.pos = pos + float4(-.03, .03, 0, 0); //mvp.proj * v2;
  output.uv = float2(0.f, 1.f);
  stream.Append(output);

  output.pos = pos + float4(.03, -.03, 0, 0); //mvp.proj * v3;
  output.uv = float2(1.f, 0.f);
  stream.Append(output);

  output.pos = pos + float4(.03, .03, 0, 0); //mvp.proj * v4;
  output.uv = float2(1.f, 1.f);
  stream.Append(output);
}
