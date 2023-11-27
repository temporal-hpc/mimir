#include "generics.slang"

interface IColor
{
    float4 getColor();
};

struct ColorFloat4 : IColor
{
    VectorInput<float,4> colors;
    float4 getColor() { return colors.data; }
};

struct ColorInt1 : IColor
{
    static const float4 colors[] = {
        {27/255.f,158/255.f,119/255.f,1},
        {217/255.f,95/255.f,2/255.f,1},
        {117/255.f,112/255.f,179/255.f,1},
        {231/255.f,41/255.f,138/255.f,1}
    };
    VectorInput<int,1> input : COLOR;
    float4 getColor()
    {
        return colors[input.data.x];
    }
};

struct ColorFloat2 : IColor
{
    VectorInput<float,2> input : COLOR;
    float4 getColor()
    {
        float2 data = convertFloat<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorDouble2 : IColor
{
    VectorInput<double,2> input : COLOR;
    float4 getColor()
    {
        float2 data = convertDouble<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorHalf2 : IColor
{
    VectorInput<half,2> input : COLOR;
    float4 getColor()
    {
        float2 data = convertHalf<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorInt2 : IColor
{
    VectorInput<int,2> input : COLOR;
    float4 getColor()
    {
        float2 data = convertInt<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorFloat3 : IColor
{
    VectorInput<float,3> input : COLOR;
    float4 getColor()
    {
        return getColorVec3(input.data);
    }
};

struct ColorDouble3 : IColor
{
    VectorInput<double,3> input : COLOR;
    float4 getColor()
    {
        float3 data = convertDouble<3>(input.data);
        return getColorVec3(data);
    }
};

struct ColorHalf3 : IColor
{
    VectorInput<half,3> input : COLOR;
    float4 getColor()
    {
        float3 data = convertHalf<3>(input.data);
        return getColorVec3(data);
    }
};

struct ColorInt3 : IColor
{
    VectorInput<int,3> input : COLOR;
    float4 getColor()
    {
        float3 data = convertInt<3>(input.data);
        return getColorVec3(data);
    }
};

float4 getColorVec2(float2 data) { return float4(data, 0, 1); }
float4 getColorVec3(float3 data) { return float4(data, 1); }