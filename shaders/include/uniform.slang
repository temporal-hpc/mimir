struct ModelViewProjection
{
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

struct SceneParams
{
  int3 extent;
  float4 bg_color;
  float depth;
};

struct PrimitiveParams
{
  float4 color;
  float size;
};
