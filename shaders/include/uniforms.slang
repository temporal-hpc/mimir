struct ModelViewProjection
{
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4x4 all;
    float4x4 inv_model;
    float4x4 inv_view;
};

struct SceneParams
{
    float4 background_color;
    int3 extent;
    int2 resolution;
    float3 camera_pos;
    float3 light_pos;
    float3 light_color;
    float3 specular_color;
    float specular_power;
    float ambient_strength;
};

struct ViewParams
{
    float4 default_color;
    float default_size;
    float linewidth;
    float antialias;
};

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> scene;
ConstantBuffer<ViewParams> view;

[[vk::binding(5)]] StructuredBuffer<float4> colorbuf;