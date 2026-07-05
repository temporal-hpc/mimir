import uniforms;

interface ITexcoord
{
    float2 getTexcoord();
};

struct TexcoordDefault : ITexcoord
{
    float2 getTexcoord() { return float2(0, 0); }
};

struct TexcoordFloat2 : ITexcoord
{
    float2 data : TEXCOORD;
    float2 getTexcoord() { return data; }
};
