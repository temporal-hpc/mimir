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

    float4 default_color;
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

/*struct ColorInt1 : IColor
{
    static const float3 colors[] = {
        {49,130,189},
        {49,163,84},
        {158,202,225},
    };
    int input : COLOR;
    float4 default_color;
    float4 getColor()
    {
        float3 c = colors[input];
        c.x /= 255.f;
        c.y /= 255.f;
        c.z /= 255.f;
        return float4(c, 1);
    }
};*/

struct ColorFloat1 : IColor
{
    float input : COLOR;
    float4 getColor()
    {
        float c = input;
        return float4(c,c,c,1);
    }
};

struct ColorInt1 : IColor
{
    int input : COLOR;
    float4 getColor()
    {
        float c = float(input); // / view.element_count;
        return float4(c, c, c, 1);
    }
};
