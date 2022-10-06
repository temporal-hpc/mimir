import uniform;

[[vk::binding(2)]] cbuffer ColorParamsUBO
{
  ColorParams colors;
};

struct FragmentInput
{
  float4 pos   : SV_Position;
  float4 color : COLOR;
};

float4 main(FragmentInput input) : SV_Target
{
  return colors.edge_color;
}
