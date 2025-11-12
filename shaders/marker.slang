import uniforms;

import position;  // IPosition
import color; // IColor
import size; // ISize
import styles; // IStyle
import shapes; // IShape
import rotation; // IRotation

struct VertexOutput
{
    float4 center : POSITION;
    float4 color  : COLOR;
    float size    : PSIZE;
    float2 rotation;
};

struct GeometryOutput
{
    float4 pos   : SV_Position;
    float2 uv    : TEXCOORD;
    float4 color : COLOR;
    float size   : PSIZE;
    float2 rotation;
};

struct FragmentOutput
{
    float4 color : SV_Target;
    //float depth  : SV_Depth;
};

[shader("vertex")]
VertexOutput vertexMain<P: IPosition, C: IColor, S: ISize>(
    P in_pos   : POSITION,
    C in_color : COLOR,
    S in_size  : PSIZE)
{
    OrientationDefault in_angle;
    in_angle.setDefault(0);

    VertexOutput output;
    output.center   = in_pos.getPosition();
    output.color    = in_color.getColor();
    output.size     = in_size.getSize();
    output.rotation = in_angle.getValue();
    return output;
}

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain<G: IMarkerGeometry>(
    point VertexOutput input[1],
    inout TriangleStream<GeometryOutput> stream)
{
    float4 pivot_world = mul(input[0].center, mvp.model);
    float4 pivot_view = mul(pivot_world, mvp.view);

    float sz = input[0].size;
    float4 offsets[4] = { { -sz, -sz }, { -sz, +sz }, { +sz, -sz }, { +sz, +sz } };
    float2 mapping[4] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
    G geometry;

    GeometryOutput output;
    output.size = sz;
    output.color = input[0].color;
    output.rotation = input[0].rotation;
    // Add point coordinates
    for (int i = 0; i < 4; ++i)
    {
        float4 vertex = pivot_view + geometry.getVertex(offsets[i]);
        output.pos = mul(vertex, mvp.proj);
        output.uv = mapping[i];
        stream.Append(output);
    }
}

[shader("fragment")]
FragmentOutput fragmentMain<S: IStyle, M: MarkerShape>(GeometryOutput input)
{
    float2 p = input.uv - float2(0.5);
    float2 r = input.rotation;
    p = float2(r.x * p.x - r.y * p.y, r.y * p.x + r.x * p.y);

    M shape;
    float distance = shape.apply(p * getMarkerSizePx(input.size), input.size);

    S style;
    float4 color = style.apply(distance, view.linewidth, view.antialias, input.color, scene.background_color);

    FragmentOutput out;
    out.color = color;
    return out;
}

float getMarkerSizePx(float size)
{
    return SQRT_2 * size + 2 * (view.linewidth + 1.5 * view.antialias);
}