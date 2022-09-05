struct FragmentInput
{
  float4 pos    : SV_Position;
  float4 color  : COLOR;
  float3 normal : NORMAL;
};

float3 lambert(float3 normal, float3 light, float3 color)
{
  return color * max(dot(normalize(normal), normalize(light)), 0);
}

float4 main(FragmentInput input) : SV_Target
{
  float3 light_front = float3(0, 0, 1);
  float3 light_up = float3(0, 1, 0);
  float3 light_color = input.color.xyz;

  float front_bias = .9;
  float3 ambient_light = float3(.1);
  float3 color_front = front_bias * lambert(input.normal, light_front, light_color);
  float3 color_up = (1 - front_bias) * lambert(input.normal, light_up, light_color);

  return float4(ambient_light + color_front + color_up, input.color.w);
}
