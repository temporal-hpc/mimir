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
  float3 center : POSITION;
  float4 color  : COLOR;
};

struct GeometryOutput
{
  float4 pos    : SV_Position;
  float4 color  : COLOR;
  float3 normal : NORMAL;
};

void addFace(inout TriangleStream<GeometryOutput> stream, float4 color,
  float4 center, float4 shift, float4 dy, float4 dx, float3 n)
{
  // Emit a primitive only if the sign of the dot product is positive
  float4 normal = mul(mvp.view * mvp.model, float4(n, 0.0));
  if (dot(-center.xyz, normal.xyz) > 0)
  {
    GeometryOutput output;
    output.color = color;
    output.normal = normal.xyz;

    output.pos = (center + shift) + (dx - dy);
    stream.Append(output);

    output.pos = (center + shift) + (-dx - dy);
    stream.Append(output);

    output.pos = (center + shift) + (dx + dy);
    stream.Append(output);

    output.pos = (center + shift) + (-dx + dy);
    stream.Append(output);
  }
}

[maxvertexcount(12)]
void main(point GeometryInput input[1], inout TriangleStream<GeometryOutput> stream)
{
  float4 color = input[0].color;
  float4 center_pos = float4(input[0].center, 1);

  float4x4 model_view = mvp.view * mvp.model;
  float4 center = mul(model_view, center_pos);
  float3 half_block = .5 * float3(1.0, 1.0, 1.0);

  float4 dx = model_view[0] * half_block.x;
  float4 dy = model_view[1] * half_block.y;
  float4 dz = model_view[2] * half_block.z;

  // In orthographic projection we have to fix our origin (center),
  // because every ray has the same direction
  if (mvp.proj[3][3] == 1)
  {
    center = float4(0.0, 0.0, -1.0, 1.0);
  }

  addFace(stream, color, center, +dx, dy, dz, float3( 1.0,  0.0,  0.0)); // Right
  addFace(stream, color, center, -dx, dz, dy, float3(-1.0,  0.0,  0.0)); // Left
  addFace(stream, color, center, +dy, dz, dx, float3( 0.0,  1.0,  0.0)); // Top
  addFace(stream, color, center, -dy, dx, dz, float3( 0.0, -1.0,  0.0)); // Bottom
  addFace(stream, color, center, +dz, dx, dy, float3( 0.0,  0.0,  1.0)); // Front
  addFace(stream, color, center, -dz, dy, dx, float3( 0.0,  0.0, -1.0)); // Back
}
