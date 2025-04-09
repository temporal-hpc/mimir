#include "arrows.slang"
import uniforms;

static const float SQRT_2 = 1.4142135623730951;
static const float PI = 3.14159265358979323846264;

struct MarkerOffsets
{
    float4 dx;
    float4 dy;
    float4 dz;
};

interface IMarkerGeometry
{
    MarkerOffsets getVertexOffsets(float size);
};

struct Marker2D : IMarkerGeometry
{
    MarkerOffsets getVertexOffsets(float size)
    {
        MarkerOffsets out;
        out.dx = mvp.all[0] / 2.0 * size;
        out.dy = mvp.all[1] / 2.0 * size;
        out.dz = mvp.all[2] / 2.0 * size;
        return out;
    }
};

struct Marker3D : IMarkerGeometry
{
    MarkerOffsets getVertexOffsets(float size)
    {
        MarkerOffsets out;
        out.dx = float4(size / 2, 0, 0, 0);
        out.dy = float4(0, size / 2, 0, 0);
        out.dz = float4(0, 0, size / 2, 0);
        return out;
    }
};

interface MarkerShape
{
    float apply(float2 p, float size);
}

struct Disc : MarkerShape
{
    float apply(float2 p, float size)
    {
        return length(p) - size/2;
    }
}

struct Square : MarkerShape
{
    float apply(float2 p, float size)
    {
        return max(abs(p.x), abs(p.y)) - size / (2 * SQRT_2);
    }
}

struct Triangle : MarkerShape
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

struct Diamond : MarkerShape
{
    float apply(float2 p, float size)
    {
        float x = SQRT_2/2 * (p.x - p.y);
        float y = SQRT_2/2 * (p.x + p.y);
        return max(abs(x), abs(y)) - size / (2 * SQRT_2);
    }
}

struct Chevron : MarkerShape
{
    float apply(float2 p, float size)
    {
        float x = 1.0 / SQRT_2 * (p.x - p.y);
        float y = 1.f / SQRT_2 * (p.x + p.y);
        float r1 = max(abs(x), abs(y)) - size / 3.0;
        float r2 = max(abs(x - size / 3.0), abs(y - size / 3.0)) - size / 3.0;
        return max(r1, -r2);
    }
}

struct Clover : MarkerShape
{
    float apply(float2 p, float size)
    {
        float t1 = -PI / 2.0;
        float2 c1 = 0.25 * float2(cos(t1), sin(t1));
        float t2 = t1 + 2 * PI / 3;
        float2 c2 = 0.25 * float2(cos(t2), sin(t2));
        float t3 = t2 + 2 * PI / 3;
        float2 c3 = 0.25 * float2(cos(t3), sin(t3));

        float r1 = length(p - c1 * size) - size / 3.5;
        float r2 = length(p - c2 * size) - size / 3.5;
        float r3 = length(p - c3 * size) - size / 3.5;
        return min(min(r1, r2), r3);
    }
}

struct Ring : MarkerShape
{
    float apply(float2 p, float size)
    {
        float r1 = length(p) - size / 2.0;
        float r2 = length(p) - size / 4.0;
        return max(r1, -r2);
    }
}

struct Tag : MarkerShape
{
    float apply(float2 p, float size)
    {
        float r1 = max(abs(p.x) - size / 2.0, abs(p.y) - size / 6.0);
        float r2 = abs(p.x - size / 1.5) + abs(p.y) - size;
        return max(r1, 0.75 * r2);
    }
}

struct Cross : MarkerShape
{
    float apply(float2 p, float size)
    {
        float x = SQRT_2 / 2.0 * (p.x - p.y);
        float y = SQRT_2 / 2.0 * (p.x + p.y);
        float r1 = max(abs(x - size / 3.0), abs(x + size / 3.0));
        float r2 = max(abs(y - size / 3.0), abs(y + size / 3.0));
        float r3 = max(abs(x), abs(y));
        return max(min(r1, r2), r3) - size / 2.0;
    }
}

struct Asterisk : MarkerShape
{
    float apply(float2 p, float size)
    {
        float x = SQRT_2 / 2.0 * (p.x - p.y);
        float y = SQRT_2 / 2.0 * (p.x + p.y);
        float r1 = max(abs(x) - size / 2.0, abs(y) - size / 10.0);
        float r2 = max(abs(y) - size / 2.0, abs(x) - size / 10.0);
        float r3 = max(abs(p.x) - size / 2.0, abs(p.y) - size / 10.0);
        float r4 = max(abs(p.y) - size / 2.0, abs(p.x) - size / 10.0);
        return min(min(r1, r2), min(r3, r4));
    }
}

struct Infinity : MarkerShape
{
    float apply(float2 p, float size)
    {
        float2 c1 = float2(+0.2125, 0.0);
        float2 c2 = float2(-0.2125, 0.0);
        float r1 = length(p - c1 * size) - size / 3.5;
        float r2 = length(p - c1 * size) - size / 7.5;
        float r3 = length(p - c2 * size) - size / 3.5;
        float r4 = length(p - c2 * size) - size / 7.5;
        return min(max(r1, -r2), max(r3, -r4));
    }
}

struct Pin : MarkerShape
{
    float apply(float2 p, float size)
    {
        float2 c1 = float2(0.0, -0.15) * size;
        float r1 = length(p - c1) - size / 2.675;
        float2 c2 = float2(+1.49, -0.80) * size;
        float r2 = length(p - c2) - 2 * size;
        float2 c3 = float2(-1.49, -0.80) * size;
        float r3 = length(p - c3) - 2 * size;
        float r4 = length(p - c1) - size / 5;
        return max(min(r1, max(max(r2, r3), -p.y)), -r4);
    }
}

struct Ellipse : MarkerShape
{
    float apply(float2 P, float size)
    {
        float2 ab = float2(size / 3.0, size / 2.0);
        float2 p = abs(P);
        if (p.x > p.y) { p = p.yx; ab = ab.yx; }
        float l = ab.y * ab.y - ab.x * ab.x;
        float m = ab.x * p.x / l;
        float n = ab.y * p.y / l;
        float m2 = m * m;
        float n2 = n * n;
        float c = (m2 + n2 - 1.0) / 3.0;
        float c3 = c * c * c;
        float q = c3 + m2 * n2 * 2.0;
        float d = c3 + m2 * n2;
        float g = m + m * n2;
        float co;

        if (d < 0.0)
        {
            float p = acos(q / c3) / 3.0;
            float s = cos(p);
            float t = sin(p) * sqrt(3.0);
            float rx = sqrt(-c * (s + t + 2.0) + m2);
            float ry = sqrt(-c * (s - t + 2.0) + m2);
            co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) / 2.0;
        }
        else
        {
            float h = 2.0 * m * n * sqrt(d);
            float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
            float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
            float rx = -s - u - c * 4.0 + 2.0 * m2;
            float ry = (s - u) * sqrt(3.0);
            float rm = sqrt(rx * rx + ry * ry);
            float p = ry / sqrt(rm - rx);
            co = (p + 2.0 * g / rm - m) / 2.0;
        }

        float si = sqrt(1.0 - co * co);
        float2 closest_point = float2(ab.x * co, ab.y * si);
        return length(closest_point - p) * sign(p.y - closest_point.y);
    }
}

struct ArrowBlock : MarkerShape
{
    float apply(float2 p, float size)
    {
        float x = p.x;
        float y = p.y;
        float r1 = abs(x) + abs(y) - size / 2;
        float r2 = max(abs(x + size / 2), abs(y)) - size / 2;
        float r3 = max(abs(x - size / 6) - size / 4, abs(y) - size / 4);
        return min(r3, max(0.75 * r1, r2));
    }
}

struct ArrowCurved : MarkerShape
{
    float apply(float2 p, float size)
    {
        float body = size;
        float head = 0.25 * body;
        return arrowCurved(p, body, head, view.linewidth, view.antialias);
    }
}

struct ArrowStealth : MarkerShape
{
    float apply(float2 p, float size)
    {
        float body = size;
        float head = 0.25 * body;
        return arrowStealth(p, body, head, view.linewidth, view.antialias);
    }
}

struct ArrowTriangle : MarkerShape
{
    float apply(float2 p, float size)
    {
        float body = size;
        float head = 0.2 * body;
        return arrowTriangle60(p, body, head, view.linewidth, view.antialias);
    }
}

struct ArrowAngle : MarkerShape
{
    float apply(float2 p, float size)
    {
        float body = size;
        float head = 0.25 * body;
        return arrowAngle60(p, body, head, view.linewidth, view.antialias);
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

struct Sphere : IShape3D
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