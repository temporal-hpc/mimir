interface IOrientation
{
    float2 getValue();
};

float2 getRotation(float o)
{
    return float2(cos(o), sin(o));
}

struct OrientationDefault : IOrientation
{
    float default_value;
    [mutating] void setDefault(float sz) { default_value = sz; }
    float2 getValue() { return getRotation(default_value); }
};

// Normal type casts do not work on template types, so we use reinterpret
struct OrientationTemplate<T> : IOrientation
{
    T input : PSIZE;

    float default_value;
    [mutating] void setDefault(float sz) { default_value = sz; }
    float2 getValue() { return getRotation(reinterpret<float>(input)); }
};

// Because reinterpret does not work when target type is smaller than source type,
// we cast over the reinterpreted double type
struct OrientationTemplateLarge<T> : IOrientation
{
    T input : PSIZE;
    float default_value;
    [mutating] void setDefault(float sz) { default_value = sz; }
    float2 getValue() { return getRotation(float(reinterpret<double>(input))); }
};

typedef OrientationTemplate<int> OrientationInt1;
typedef OrientationTemplate<float> OrientationFloat1;
typedef OrientationTemplate<half> OrientationHalf1;
typedef OrientationTemplateLarge<double> OrientationDouble1;