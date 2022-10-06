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
  float4 pos    : SV_Position;
  float4 color  : COLOR;
  float3 normal : NORMAL;
};

void addFace(inout TriangleStream<GeometryOutput> stream, float4 color,
  float4 center, float4 shift, float4 dy, float4 dx, float3 n)
{
  float4x4 model_view = mul(mvp.model, mvp.view);
  float4 normal = mul(float4(n, 0), model_view);
  // Emit a primitive only if the sign of the dot product is positive
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
  float4 pos = input[0].pos;
  float4 color = input[0].color;

  float4x4 model_view = mul(mvp.model, mvp.view);
  float4 center = mul(input[0].center, model_view);
  // In orthographic projection we have to fix our origin (center),
  // because every ray has the same direction
  if (mvp.proj[3][3] == 1)
  {
    center = float4(0, 0, -1, 1);
  }

  // Compute offsets for block (voxel) vertices
  float3 half_block = .5 * float3(.01);
  float4 bx = model_view[0] * half_block.x;
  float4 by = model_view[1] * half_block.y;
  float4 bz = model_view[2] * half_block.z;

  float4 v, shift, dx, dy, normal;
  GeometryOutput output;
  output.color = color;

  normal = mul(float4(1, 0, 0, 0), model_view);
  // Emit a primitive only if the sign of the dot product is positive
  if (dot(-center.xyz, normal.xyz) > 0)
  {
    shift = bx;
    dx = bz;
    dy = by;
  }
  else
  {
    shift = -bx;
    dx = by;
    dy = bz;
  }
  output.normal = normal.xyz;
  // Add point coordinates
  v = (center + shift) + (dx - dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (-dx - dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (dx + dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (-dx + dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);

  normal = mul(float4(0, 1, 0, 0), model_view);
  // Emit a primitive only if the sign of the dot product is positive
  if (dot(-center.xyz, normal.xyz) > 0)
  {
    shift = by;
    dx = bx;
    dy = bz;
  }
  else
  {
    shift = -by;
    dx = bz;
    dy = bx;
  }
  output.normal = normal.xyz;
  // Add point coordinates
  v = (center + shift) + (dx - dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (-dx - dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (dx + dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (-dx + dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);

  normal = mul(float4(0, 0, 1, 0), model_view);
  // Emit a primitive only if the sign of the dot product is positive
  if (dot(-center.xyz, normal.xyz) > 0)
  {
    shift = bz;
    dx = by;
    dy = bx;
  }
  else
  {
    shift = -bz;
    dx = bx;
    dy = by;
  }
  output.normal = normal.xyz;
  // Add point coordinates
  v = (center + shift) + (dx - dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (-dx - dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (dx + dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);
  v = (center + shift) + (-dx + dy);
  output.pos = mul(v, mvp.proj);
  stream.Append(output);



  /*addFace(stream, color, center, +dx, dy, dz, float3( 1.0,  0.0,  0.0)); // Right
  addFace(stream, color, center, -dx, dz, dy, float3(-1.0,  0.0,  0.0)); // Left
  addFace(stream, color, center, +dy, dz, dx, float3( 0.0,  1.0,  0.0)); // Top
  addFace(stream, color, center, -dy, dx, dz, float3( 0.0, -1.0,  0.0)); // Bottom
  addFace(stream, color, center, +dz, dx, dy, float3( 0.0,  0.0,  1.0)); // Front
  addFace(stream, color, center, -dz, dy, dx, float3( 0.0,  0.0, -1.0)); // Back*/
}
