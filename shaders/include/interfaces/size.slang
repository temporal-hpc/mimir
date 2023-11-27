#include "generics.slang"

interface ISize
{
    float getSize();
};

struct SizeInt : ISize
{
    VectorInput<int,1> input : PSIZE;
    float getSize() { return convertInt<1>(input.data).x; }
};

struct SizeFloat : ISize
{
    VectorInput<float,1> input : PSIZE;
    float getSize() { return convertFloat<1>(input.data).x; }
};

struct SizeDouble : ISize
{
    VectorInput<double,1> input : PSIZE;
    float getSize() { return convertDouble<1>(input.data).x; }
};

struct SizeHalf : ISize
{
    VectorInput<half,1> input : PSIZE;
    float getSize() { return convertHalf<1>(input.data).x; }
};