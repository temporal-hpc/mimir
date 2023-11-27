#include "generics.slang"

float4 getPosition2d(float2 data, int3 extent)
{
    return float4(2 * (data / extent.xy) - 1, 0, 1);
}

float4 getPosition3d(float3 data, int3 extent)
{
    return float4(2 * (data / extent) - 1, 1);
}

float4 getPosition4d(float4 data, int3 extent)
{
    return float4(2 * (data.xyz / extent) - 1, 1);
}

interface IPosition
{
    float4 getPosition(int3 extent);
};

struct PositionFloat2 : IPosition
{
    VectorInput<float,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertFloat<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionDouble2 : IPosition
{
    VectorInput<double,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertDouble<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionHalf2 : IPosition
{
    VectorInput<half,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertHalf<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionInt2 : IPosition
{
    VectorInput<int,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertInt<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionFloat3 : IPosition
{
    VectorInput<float,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        return getPosition3d(input.data, extent);
    }
};

struct PositionFloat4 : IPosition
{
    VectorInput<float,4> input : POSITION;
    float4 getPosition(int3 extent)
    {
        return getPosition4d(input.data, extent);
    }
};

struct PositionDouble3 : IPosition
{
    VectorInput<double,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float3 data = convertDouble<3>(input.data);
        return getPosition3d(data, extent);
    }
};

struct PositionHalf3 : IPosition
{
    VectorInput<half,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float3 data = convertHalf<3>(input.data);
        return getPosition3d(data, extent);
    }
};

struct PositionInt3 : IPosition
{
    VectorInput<int,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float3 data = convertInt<3>(input.data);
        return getPosition3d(data, extent);
    }
};