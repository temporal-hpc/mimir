import uniforms;

import position;  // IPosition
import color; // IColor
import size; // ISize
import antialias; // IAntialiased
import shapes; // IShape
import rotation; // IRotation

#define M_PI 3.1415926535897932384626433832795

float getMarkerSizePx(float size)
{
    return SQRT_2 * size + 2 * (v_linewidth + 1.5 * u_antialias);
}

static const float v_linewidth = 0;
static const float u_antialias = 0;

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
    float ele_size;
    //float3 normal : NORMAL;
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
    output.center   = mul(in_pos.getPosition(), mvp.all);
    output.color    = in_color.getColor();
    output.size     = in_size.getSize();
    output.rotation = in_angle.getValue();
    return output;
}

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain<M: IMarker>(
    point VertexOutput input[1],
    inout TriangleStream<GeometryOutput> stream)
{
    float4 center = input[0].center;

    GeometryOutput output;
    // TODO: Add antialias and linewidth to output.size (something like getMarkerSizePx)
    output.size     = input[0].size;
    output.ele_size = input[0].size;

    M marker = {};
    MarkerOffsets offsets = marker.getVertexOffsets(output.ele_size);
    float4 dx = offsets.dx;
    float4 dy = offsets.dy;

    float4 vertex_offsets[4] = {(-dx-dy), (+dx-dy), (-dx+dy), (+dx+dy)};
    float2 texcoords[4]      = {{0,0}, {1,0}, {0,1}, {1,1}};

    output.color    = input[0].color;
    output.rotation = input[0].rotation;
    //output.normal = normal.xyz;

    // Add point coordinates
    for (int i = 0; i < 4; ++i)
    {
        output.pos = center + vertex_offsets[i];
        output.uv  = texcoords[i];
        stream.Append(output);
    }
}

[shader("fragment")]
FragmentOutput fragmentMain<S: IShape2D>(GeometryOutput input)
{
    FragmentOutput out;

    float2 p = input.uv - float2(0.5);
    float2 rot = input.rotation;
    p = float2(rot.x * p.x - rot.y * p.y, rot.y * p.x + rot.x * p.y);

    S shape = {};
    float distance = shape.apply(p * input.size, input.ele_size);

    Filled aa;
    out.color = aa.apply(distance, v_linewidth, u_antialias, input.color, scene.background_color);
    //float4x4 view_proj = mul(mvp.view, mvp.proj);
    //float4 clip_pos = mul(view_proj, float4(1, 1, 1, 1));
    //out.depth = clip_pos.z / clip_pos.w;
    //out.depth = 1 - out.color.w;
    return out;
}
