import uniform;

[[vk::binding(0)]] ConstantBuffer<ModelViewProjection> mvp;
[[vk::binding(1)]] ConstantBuffer<SceneParams> scene;
[[vk::binding(2)]] ConstantBuffer<PrimitiveParams> primitive;

struct VertexInput
{
  [[vk::location(0)]] float2 pos : POSITION;
};

struct VertexData
{
  float4 pos    : SV_Position;
  float4 center : POSITION;
  //float4 color  : COLOR;
  //float marker_size;
};

[shader("vertex")]
VertexData vertex2dMain(VertexInput input)
{
  float4x4 view_proj = mul(mvp.view, mvp.proj);

  VertexData output;
  output.center = float4(2 * (float3(input.pos, 1) / scene.extent) - 1, 1);
  output.pos    = mul(output.center, view_proj);
  //output.color  = float4(0, 0, 1, 1);
  //output.marker_size = 10.f;

  return output;
}

struct VertexInput3d
{
  [[vk::location(0)]] float3 pos : POSITION;
};

[shader("vertex")]
VertexData vertex3dMain(VertexInput3d input)
{
  float4x4 view_proj = mul(mvp.view, mvp.proj);

  VertexData output;
  output.center = float4(2 * (input.pos / scene.extent) - 1, 1);
  output.pos    = mul(output.center, view_proj);
  //output.color  = float4(0, 0, 1, 1);
  //output.marker_size = 10.f;

  return output;
}

struct MarkerData
{
  float4 pos   : SV_Position;
  float2 uv    : TEXCOORD;
  //float4 color : COLOR;
  //float3 normal : NORMAL;
};

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain(point VertexData input[1], inout TriangleStream<MarkerData> stream)
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

  MarkerData output;
  //output.color = input[0].color;
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


static const float SQRT_2 = 1.4142135623730951;
static const float u_antialias = 1;
static const float v_linewidth = 1;

float4 outline(
  float distance, // Signed distance to line
  float linewidth, // Stroke line width
  float antialias, // Stroke antialiased area
  float4 stroke, // Stroke color
  float4 fill) // Fill color
{
  float t = linewidth / 2.0 - antialias;
  float signed_distance = distance;
  float border_distance = abs(signed_distance) - t;
  float alpha = border_distance / antialias;
  alpha = exp(-alpha * alpha);
  if( border_distance < 0.0 )
  return stroke;
  else if( signed_distance < 0.0 )
  return lerp(fill, stroke, sqrt(alpha));
  else
  return float4(stroke.rgb, stroke.a * alpha);
}

float disc(float2 p, float size)
{
  return length(p) - size/2;
}

[shader("fragment")]
float4 fragmentMain(MarkerData input) : SV_Target
{
  float2 center_coords = 2 * input.uv - 1;
  if (dot(center_coords, center_coords) > 1) discard;
  return primitive.color;
  /*float2 p = input.uv - float2(0.5);
  //p = float2(v_rotation.x*p.x - v_rotation.y*p.y,
  //           v_rotation.y*p.x + v_rotation.x*p.y);
  float point_size = SQRT_2*primitive.size  + 2 * (v_linewidth + 1.5*u_antialias);
  float distance = disc(p*point_size, primitive.size);
  return outline(distance, v_linewidth, u_antialias, scene.bg_color, primitive.color);*/
}
