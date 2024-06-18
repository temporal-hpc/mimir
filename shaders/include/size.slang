import uniforms;

interface ISize
{
    float getSize();
};

struct SizeDefault : ISize
{
    float getSize() { return view.default_size; }
};

// Normal type casts do not work on template types, so we use reinterpret
struct SizeTemplate<T> : ISize
{
    T input : PSIZE;

    float getSize() { return reinterpret<float>(input); }
};

// Because reinterpret does not work when target type is smaller than source type,
// we cast over the reinterpreted double type
struct SizeTemplateLarge<T> : ISize
{
    T input : PSIZE;
    float getSize() { return float(reinterpret<double>(input)); }
};

typedef SizeTemplate<int> SizeInt1;
typedef SizeTemplate<float> SizeFloat1;
typedef SizeTemplate<half> SizeHalf1;
typedef SizeTemplateLarge<double> SizeDouble1;
