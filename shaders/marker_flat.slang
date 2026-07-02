// Flat 2D marker shader — cheap alternative to marker.slang.
//
// Uses native point sprites (gl_PointSize / gl_PointCoord) instead of a geometry
// shader quad expansion.  No ray-sphere intersection, no lighting, no custom depth
// write.
//
// The fragment stage replicates datoviz's disc marker exactly (graphics_marker.frag,
// DISC/FILLED, linewidth 0, angle 0): the SDF treats the sprite as `size + antialias`
// pixels wide, so the opaque disc is size²/(size+antialias) px with a ~1 px gaussian
// skirt.  This makes a given size value produce the SAME on-screen disc as a datoviz
// marker of that size — required for fair benchmark comparisons.
//
// Enabled via MarkerOptions::RenderMode::Flat2D.  All other programs that do not set
// that field continue to use marker.slang (Sphere3D) unchanged.

import uniforms;
import position;
import color;
import size;

// Matches datoviz's constants.glsl (ANTIALIAS, "antialias = 1 everywhere").
static const float ANTIALIAS = 1.0;

// Vertex stage outputs.  SV_PointSize sets gl_PointSize; it is a GPU built-in and
// is NOT interpolated to fragments, so the size is also passed as a regular varying
// (PSIZE) for the fragment SDF.
struct VSOut
{
    float4 pos   : SV_Position;
    float4 color : COLOR;
    float  szv   : PSIZE;
    float  sz    : SV_PointSize;
};

// Fragment stage input — only the interpolated user varyings (no SV_PointSize).
struct FSIn
{
    float4 color : COLOR;
    float  szv   : PSIZE;
};

// gl_PointCoord: slang has no system-value semantic for the point-sprite coordinate
// (HLSL has no point sprites), so read the SPIR-V builtin directly.
float2 pointCoord()
{
    return spirv_asm {
        result:$$float2 = OpLoad builtin(PointCoord:float2);
    };
}

[shader("vertex")]
VSOut vertexMain<P: IPosition, C: IColor, S: ISize>(
    P in_pos   : POSITION,
    C in_color : COLOR,
    S in_size  : PSIZE)
{
    VSOut o;
    o.pos   = mul(in_pos.getPosition(), mvp.all);
    o.color = in_color.getColor();
    o.szv   = in_size.getSize();
    o.sz    = o.szv;
    return o;
}

[shader("fragment")]
float4 fragmentMain(FSIn input) : SV_Target
{
    // Point coord is in [0,1]; recenter to [-0.5, 0.5] like datoviz's P.
    float2 p = pointCoord() - float2(0.5, 0.5);

    // datoviz corner discard (graphics_marker.frag, eps = 5e-3).
    const float eps = 5e-3;
    if (abs(p.x) >= 0.5 - eps || abs(p.y) >= 0.5 - eps) { discard; }

    // datoviz marker_disc SDF over a sprite treated as (size + antialias) px wide:
    // d < 0 inside the opaque disc, gaussian alpha falloff outside (filled(), the
    // linewidth-0 path where t = -antialias).
    float sz = input.szv;
    float d  = length(p * (sz + ANTIALIAS)) - sz / 2.0;

    float4 color = input.color;
    if (d >= 0.0)
    {
        float border = (d + ANTIALIAS) / ANTIALIAS;
        color.a *= exp(-border * border);
    }
    return color;
}
