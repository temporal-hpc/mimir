#include "generics.slang"

interface ISize
{
    [mutating] void setDefault(float sz);
    float getSize();
};

struct SizeDefault : ISize
{
    float default_size;
    [mutating] void setDefault(float sz) { default_size = sz; }
    float getSize() { return default_size; }
};

struct SizeTemplate<T> : ISize
{
    T input : PSIZE;
    float default_size;
    [mutating] void setDefault(float sz) { default_size = sz; }
    float getSize() { return reinterpret<float>(input).x; }
};

typedef SizeTemplate<int> SizeInt1;
typedef SizeTemplate<float> SizeFloat1;
typedef SizeTemplate<half> SizeHalf1;
// TODO: This may not work due to target type being smaller than source type
//typedef SizeTemplate<double> SizeDouble;
