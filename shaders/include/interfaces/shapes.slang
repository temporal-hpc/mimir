static const float SQRT_2 = 1.4142135623730951;

interface IShape2D
{
    float apply(float2 p, float size);
}

struct DiscShape : IShape2D
{
    float apply(float2 p, float size)
    {
        return length(p) - size/2;
    }
}

struct SquareShape : IShape2D
{
    float apply(float2 p, float size)
    {
        return max(abs(p.x), abs(p.y)) - size / (2 * SQRT_2);
    }
}

struct TriangleShape : IShape2D
{
    float apply(float2 p, float size)
    {
        float x = SQRT_2/2 * (p.x - p.y);
        float y = SQRT_2/2 * (p.x + p.y);
        float r1 = max(abs(x), abs(y)) - size / (2 * SQRT_2);
        float r2 = p.y;
        return max(r1, r2);
    }
}

struct DiamondShape : IShape2D
{
    float apply(float2 p, float size)
    {
        float x = SQRT_2/2 * (p.x - p.y);
        float y = SQRT_2/2 * (p.x + p.y);
        return max(abs(x), abs(y)) - size / (2 * SQRT_2);
    }
}

interface IShape3D
{
    float2 apply(
        float3 ro, // Ray origin
        float3 rd, // Ray direction
        float3 ce, // Center
        float sz // size
    );
}

struct SphereShape : IShape3D
{
    float2 apply(float3 ro, float3 rd, float3 center, float sz)
    {
        float3 oc = ro - center;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - sz * sz;
        float h = b*b - c;
        return (h<0)? -1 : -b - sqrt(h);
    }
};