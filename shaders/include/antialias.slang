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
        if (border_distance < 0.0) // Within linestroke
            return stroke;
        else if (border_distance > (linewidth / 2 + antialias)) // Outside shape
        {
            discard;
            return float4(0);
        }
        else // Inside shape
            return float4(stroke.rgb, stroke.a * alpha);
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
        if (border_distance < 0.0) // Within linestroke
            return fill;
        else if (signed_distance < 0.0) // Inside shape
            return fill;
        else if (border_distance > (linewidth / 2 + antialias)) // Outside shape
        {
            discard;
            return float4(0); // return float4(1,0,0,1);
        }
        else // Line stroke exterior border
            return float4(fill.rgb, alpha * fill.a);
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
        if (border_distance < 0.0) // Within linestroke
            return stroke;
        else if(signed_distance < 0.0) // Inside shape
            return lerp(fill, stroke, sqrt(alpha));
        else if (border_distance > (linewidth / 2 + antialias)) // Outside shape
        {
            discard;
            return float4(0);
        }
        else // Line stroke exterior border
            return float4(stroke.rgb, stroke.a * alpha);
    }
};