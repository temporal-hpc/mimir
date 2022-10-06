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

struct SceneParams
{
  int3 extent;
  float depth;
};

struct ColorParams
{
  float4 point_color;
  float4 edge_color;
};
