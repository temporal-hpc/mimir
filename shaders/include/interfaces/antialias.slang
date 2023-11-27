interface IAntialiased
{
    float4 apply(
        float distance,  // Signed distance to line
        float linewidth, // Stroke line width
        float antialias, // Stroke antialiased area
        float4 stroke,   // Stroke color
        float4 fill      // Fill color
    );
}

struct Stroke : IAntialiased
{
    // Fill is unused
    float4 apply(float distance, float linewidth, float antialias, float4 stroke, float4 fill)
    {
        float t = linewidth / 2.0 - antialias;
        float signed_distance = distance;
        float border_distance = abs(signed_distance) - t;
        float alpha = border_distance / antialias;
        alpha = exp(-alpha * alpha);
        if (border_distance < 0.0) return stroke;
        else                       return float4(stroke.rgb, stroke.a * alpha);
    }
}

struct Filled : IAntialiased
{
    // Stroke is unused
    float4 apply(float distance, float linewidth, float antialias, float4 fill, float4 stroke)
    {
        float t = linewidth / 2.0 - antialias;
        float signed_distance = distance;
        float border_distance = abs(signed_distance) - t;
        float alpha = border_distance / antialias;
        alpha = exp(-alpha * alpha);
        if (border_distance < 0.0)      return fill;
        else if (signed_distance < 0.0) return fill;
        else                            return float4(fill.rgb, alpha * fill.a);
    }
}

struct Outline : IAntialiased
{
    float4 apply(float distance, float linewidth, float antialias, float4 stroke, float4 fill)
    {
        float t = linewidth / 2.0 - antialias;
        float signed_distance = distance;
        float border_distance = abs(signed_distance) - t;
        float alpha = border_distance / antialias;
        alpha = exp(-alpha * alpha);
        if (border_distance < 0.0)     return stroke;
        else if(signed_distance < 0.0) return lerp(fill, stroke, sqrt(alpha));
        else                           return float4(stroke.rgb, stroke.a * alpha);
    }
};