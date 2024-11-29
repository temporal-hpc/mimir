import uniforms;

interface IColor
{
    float4 getColor();
};

struct ColorDefault : IColor
{
    float4 default_color;
    float4 getColor() { return view.default_color; }
};

struct ColorTemplate<T, let N:int> : IColor
{
    vector<T,N> data : COLOR;

    float4 getColor()
    {
        float4 rgba = {0,0,0,1};
        [unroll]
        for (int i = 0; i < N; ++i) {
            if (sizeof(T) <= sizeof(float)) { rgba[i] = reinterpret<float>(data[i]); }
            else { rgba[i] = float(reinterpret<double>(data[i])); }
        }
        return rgba;
    }
};

typedef ColorTemplate<float,2> ColorFloat2;
typedef ColorTemplate<float,3> ColorFloat3;
typedef ColorTemplate<float,4> ColorFloat4;

typedef ColorTemplate<half,2> ColorHalf2;
typedef ColorTemplate<half,3> ColorHalf3;
typedef ColorTemplate<half,4> ColorHalf4;

typedef ColorTemplate<int,2> ColorInt2;
typedef ColorTemplate<int,3> ColorInt3;
typedef ColorTemplate<int,4> ColorInt4;

typedef ColorTemplate<double,2> ColorDouble2;
typedef ColorTemplate<double,3> ColorDouble3;
typedef ColorTemplate<double,4> ColorDouble4;

struct ColorFloat1 : IColor
{
    float v;
    float4 getColor() { return float4(v, v, v, 1); }
};

// struct ColorInt1 : IColor
// {
//     int v;
//     float4 getColor() { return float4(v, v, v, 1); }
// };

struct ColorInt1 : IColor
{
    static const float3 colors[] = {
        {49,130,189},
        {49,163,84},
        {158,202,225},
    };
    int input : COLOR;
    float4 getColor()
    {
        float3 c = colors[input];
        c.x /= 255.f;
        c.y /= 255.f;
        c.z /= 255.f;
        return float4(c, 1);
    }
};