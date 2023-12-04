float lineDistance(float2 p, float2 p1, float2 p2)
{
    float2 center = (p1 + p2) * 0.5;
    float len = length(p2 - p1);
    float2 dir = (p2 - p1) / len;
    float2 rel_p = p - center;
    return dot(rel_p, float2(dir.y, -dir.x));
}

float segmentDistance(float2 p, float2 p1, float2 p2)
{
    float2 center = (p1 + p2) * 0.5;
    float len = length(p2 - p1);
    float2 dir = (p2 - p1) / len;
    float2 rel_p = p - center;
    float dist1 = abs(dot(rel_p, float2(dir.y, -dir.x)));
    float dist2 = abs(dot(rel_p, dir)) - 0.5 * len;
    return max(dist1, dist2);
}

float4 inscribedCircle(float2 p1, float2 p2, float radius)
{
    float q = length(p2 - p1);
    float2 m = (p1+p2)/2.0;
    float sqrt_q = sqrt(radius * radius - (q*q/4));
    float2 d = float2(sqrt_q * (p1.y - p2.y) / q, sqrt_q * (p2.x - p1.x) / q);
    return float4(m+d, m-d);
}

float arrowCurved(float2 uv, float body, float head, float linewidth, float antialias)
{
    float w = linewidth/2.0 + antialias;
    float2 start = -float2(body/2.0, 0);
    float2 end   = +float2(body/2.0, 0);
    float height = 0.5;

    float2 p1 = end - head * float2(+1.0, +height);
    float2 p2 = end - head * float2(+1.0, -height);
    float2 p3 = end;

    float2 c1 = inscribedCircle(p1, p3, 1.25 * body).zw;
    float d1 = length(uv - c1) - 1.25 * body;
    float2 c2 = inscribedCircle(p2, p3, 1.25 * body).xy;
    float d2 = length(uv - c2) - 1.25 * body;
    float2 c3 = inscribedCircle(p1, p2, max(body - head, 1.0 * body)).xy;
    float d3 = length(uv - c3) - max(body - head, 1.0 * body);

    float d4 = segmentDistance(uv, start, end - float2(linewidth, 0));

    if (uv.y > +(2.0 * head + antialias)) return 1000.0;
    if (uv.y < -(2.0 * head + antialias)) return 1000.0;
    if (uv.x < -(body / 2.0 + antialias)) return 1000.0;
    if (uv.x > c1.x) return 1000.0;

    return min(d4, -min(d3, min(d1, d2)));
}

float arrowStealth(float2 uv, float body, float head, float linewidth, float antialias)
{
    float w = linewidth/2.0 + antialias;
    float2 start = -float2(body/2.0, 0);
    float2 end   = +float2(body/2.0, 0);
    float height = 0.5;
    float2 end1 = end - float2(3.0*head/4.0, 0);

    // Head: 4 lines
    float d1 = lineDistance(uv, end - head * float2(+1.0, -height), end);
    float d2 = lineDistance(uv, end - head * float2(+1.0, -height), end1);
    float d3 = lineDistance(uv, end - head * float2(+1.0, +height), end);
    float d4 = lineDistance(uv, end - head * float2(+1.0, +0.5), end1);
    // Body: 1 segment
    float d5 = segmentDistance(uv, start, end - float2(linewidth, 0));

    return min(d5, max(max(-d1, d3), -max(-d2, d4)));
}

float arrowTriangle(float2 uv, float body, float head,
    float height, float linewidth, float antialias)
{
    float w = linewidth / 2.0 + antialias;
    float2 start = -float2(body/2.0, 0);
    float2 end   = +float2(body/2.0, 0);

    float d1 = lineDistance(uv, end, end - head * float2(+1.0, -height));
    float d2 = lineDistance(uv, end - head * float2(+1.0, +height), end);
    float d3 = uv.x - end.x + head;

    float d4 = segmentDistance(uv, start, end - float2(linewidth, 0));

    return min(max(max(d1, d2), -d3), d4);
}

float arrowTriangle90(float2 uv, float body, float head, float linewidth, float antialias)
{
    return arrowTriangle(uv, body, head, 1.0, linewidth, antialias);
}

float arrowTriangle60(float2 uv, float body, float head, float linewidth, float antialias)
{
    return arrowTriangle(uv, body, head, 0.5, linewidth, antialias);
}

float arrowTriangle30(float2 uv, float body, float head, float linewidth, float antialias)
{
    return arrowTriangle(uv, body, head, 0.25, linewidth, antialias);
}

float arrowAngle(float2 uv, float body, float head,
    float height, float linewidth, float antialias)
{
    float d;
    float w = linewidth / 2.0 + antialias;
    float2 start = -float2(body/2.0, 0);
    float2 end   = +float2(body/2.0, 0);

    if (uv.x > body / 2.0)
    {
        float d1 = lineDistance(uv, end, end - head * float2(+1.0, -height));
        float d2 = lineDistance(uv, end - head * float2(+1.0, +height), end);
        float d3 = end.x - uv.x;
        d = max(max(d1, d2), d3);
    }
    else
    {
        float d1 = segmentDistance(uv, end - head * float2(+1.0, -height), end);
        float d2 = segmentDistance(uv, end - head * float2(+1.0, +height), end);
        float d3 = segmentDistance(uv, start, end - float2(linewidth, 0));
        d = min(min(d1, d2), d3);
    }
    return d;
}

float arrowAngle90(float2 uv, float body, float head, float linewidth, float antialias)
{
    return arrowAngle(uv, body, head, 1.0, linewidth, antialias);
}

float arrowAngle60(float2 uv, float body, float head, float linewidth, float antialias)
{
    return arrowAngle(uv, body, head, 0.5, linewidth, antialias);
}

float arrowAngle30(float2 uv, float body, float head, float linewidth, float antialias)
{
    return arrowAngle(uv, body, head, 0.25, linewidth, antialias);
}