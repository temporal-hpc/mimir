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
    int2 resolution;
    float3 camera_pos;
};

struct PrimitiveParams
{
    float4 color;
    float size;
    float depth;
};
