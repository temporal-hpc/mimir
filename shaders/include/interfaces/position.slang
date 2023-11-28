#include "generics.slang"

interface IPosition
{
    float4 getPosition(int3 extent);
};

struct PositionTemplate<T, let N:int> : IPosition
{
    vector<T,N> data : POSITION;

    float4 getPosition(int3 extent)
    {
        float4 pos = {0,0,0,1};
        [unroll]
        for (int i = 0; i < N; ++i)
        {
            float r = reinterpret<float>(data[i]) / extent[i];
            pos[i] = 2 * r - 1;
        }
        pos.w = 1;
        return pos;
    }
};

struct PositionTemplateLarge<T, let N:int> : IPosition
{
    vector<T,N> data : POSITION;

    float4 getPosition(int3 extent)
    {
        float4 pos = {0,0,0,1};
        [unroll]
        for (int i = 0; i < N; ++i)
        {
            float r = float(reinterpret<double>(data[i])) / extent[i];
            pos[i] = 2 * r - 1;
        }
        pos.w = 1;
        return pos;
    }
};

typedef PositionTemplate<float,2> PositionFloat2;
typedef PositionTemplate<float,3> PositionFloat3;
typedef PositionTemplate<float,4> PositionFloat4;

typedef PositionTemplate<half,2> PositionHalf2;
typedef PositionTemplate<half,3> PositionHalf3;
typedef PositionTemplate<half,4> PositionHalf4;

typedef PositionTemplate<int,2> PositionInt2;
typedef PositionTemplate<int,3> PositionInt3;
typedef PositionTemplate<int,4> PositionInt4;

typedef PositionTemplateLarge<double,2> PositionDouble2;
typedef PositionTemplateLarge<double,3> PositionDouble3;
typedef PositionTemplateLarge<double,4> PositionDouble4;
