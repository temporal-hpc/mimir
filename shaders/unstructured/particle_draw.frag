struct FragmentInput
{
  float4 pos : SV_Position;
  float2 center_pos : POSITION1;
  float4 color : COLOR;
  float marker_size;
  //float line_width;
};

static const float SQRT_2 = 1.4142135623730951;
static const float u_antialias = 1;
static const float4 v_bg_color = float4(1, 1, 1, 1);

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

float4 main(FragmentInput input) : SV_Target
{
  float2 point_coord = (input.pos - input.center_pos) / input.marker_size + 0.5f;
  float2 center_coords = 2.f * point_coord - 1.f;
  float r = dot(center_coords, center_coords);
  if (r > 1.f) discard;
  return input.color;
  /*float2 p = point_coord - float2(0.5);
  //p = float2(v_rotation.x*p.x - v_rotation.y*p.y,
  //           v_rotation.y*p.x + v_rotation.x*p.y);
  float point_size = SQRT_2*v_size  + 2 * (v_linewidth + 1.5*u_antialias);
  float distance = disc(p*point_size, v_size);
  return outline(distance, v_linewidth, u_antialias, input.color, v_bg_color);*/
}
