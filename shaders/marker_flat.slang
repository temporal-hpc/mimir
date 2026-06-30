// Flat 2D marker shader — cheap alternative to marker.slang.
//
// Uses native point sprites (gl_PointSize / gl_PointCoord) instead of a geometry
// shader quad expansion.  The fragment stage clips each point to a disc using only
// the point-coordinate distance — no ray-sphere intersection, no lighting, no custom
// depth write.  This matches the GPU cost of a datoviz-style renderer.
//
// Enabled via MarkerOptions::RenderMode::Flat2D.  All other programs that do not set
// that field continue to use marker.slang (Sphere3D) unchanged.

import uniforms;
import position;
import color;
import size;

// Vertex stage outputs.  SV_PointSize sets gl_PointSize; it is a GPU built-in and
// is NOT interpolated to fragments, so the fragment stage uses its own input struct.
struct VSOut
{
    float4 pos   : SV_Position;
    float4 color : COLOR;
    float  sz    : SV_PointSize;
};

// Fragment stage input — only the interpolated user varyings (no SV_PointSize).
struct FSIn
{
    float4 color : COLOR;
};

[shader("vertex")]
VSOut vertexMain<P: IPosition, C: IColor, S: ISize>(
    P in_pos   : POSITION,
    C in_color : COLOR,
    S in_size  : PSIZE)
{
    VSOut o;
    o.pos   = mul(in_pos.getPosition(), mvp.all);
    o.color = in_color.getColor();
    o.sz    = in_size.getSize();
    return o;
}

[shader("fragment")]
float4 fragmentMain(FSIn input, float2 pt : SV_PointCoord) : SV_Target
{
    // pt is in [0,1]; center at (0.5, 0.5).  Discard corners outside the inscribed
    // disc (dot(p,p) > 0.25  ↔  distance from center > 0.5).
    float2 p = pt - float2(0.5, 0.5);
    if (dot(p, p) > 0.25) { discard; }
    return input.color;
}
