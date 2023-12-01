interface IColor
{
    [mutating] void setDefault(float4 c);
    float4 getColor();
};

struct ColorDefault : IColor
{
    float4 default_color;
    [mutating] void setDefault(float4 c) { default_color = c; };
    float4 getColor() { return default_color; }
};

struct ColorTemplate<T, let N:int> : IColor
{
    vector<T,N> data : COLOR;

    float4 default_color;
    [mutating] void setDefault(float4 c) { default_color = c; };
    float4 getColor()
    {
        float4 rgba = {0,0,0,1};
        [unroll]
        for (int i = 0; i < N; ++i) { rgba[i] = reinterpret<float>(data[i]); }
        return rgba;
    }
};

struct ColorTemplateLarge<T, let N:int> : IColor
{
    vector<T,N> data : COLOR;

    float4 default_color;
    [mutating] void setDefault(float4 c) { default_color = c; };
    float4 getColor()
    {
        float4 rgba = {0,0,0,1};
        [unroll]
        for (int i = 0; i < N; ++i) { rgba[i] = float(reinterpret<double>(data[i])); }
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

typedef ColorTemplateLarge<double,2> ColorDouble2;
typedef ColorTemplateLarge<double,3> ColorDouble3;
typedef ColorTemplateLarge<double,4> ColorDouble4;

struct ColorInt1 : IColor
{
    static const float4 colors[] = {
        {49,130,189,1},
        {49,163,84,1},
    };
    int input : COLOR;
    float4 default_color;
    [mutating] void setDefault(float4 c) { default_color = c; };
    float4 getColor()
    {
        float4 c = colors[input];
        c.x /= 255.f;
        c.y /= 255.f;
        c.z /= 255.f;
        return c;
    }
};
