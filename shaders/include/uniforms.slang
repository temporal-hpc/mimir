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
    float3 light_pos;
    float4 light_color;
};

struct ViewParamsOld
{
    float4 default_color;
    float default_size;
    float depth;
    int element_count;
};

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> scene;
ConstantBuffer<ViewParamsOld> view;