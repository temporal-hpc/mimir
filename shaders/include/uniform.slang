struct ModelViewProjection
{
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};

struct SceneParams
{
  float4 bg_color;
  int3 extent;
  float depth;
};

struct PrimitiveParams
{
  float4 color;
  float size;
};
