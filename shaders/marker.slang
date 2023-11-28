import uniforms

#include "interfaces/position.slang"
#include "interfaces/color.slang"
#include "interfaces/size.slang"
#include "interfaces/antialias.slang"
#include "interfaces/shapes.slang"

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> scene;
ConstantBuffer<ViewParams> view;

static const float u_antialias = 1;
static const float v_linewidth = 1;

struct VertexOutput
{
    float4 center : POSITION;
    float4 color  : COLOR;
    float size    : PSIZE;
};

struct GeometryOutput
{
    float4 pos   : SV_Position;
    float2 uv    : TEXCOORD;
    float4 color : COLOR;
    float size   : PSIZE;
    float ele_size;
    //float3 normal : NORMAL;
};

struct FragmentOutput
{
    float4 color : SV_Target;
    float depth  : SV_Depth;
};

[shader("vertex")]
VertexOutput vertexMain<P:IPosition, C:IColor, S:ISize>(P in_pos, C in_color, S in_size)
{
    in_color.setDefault(view.default_color);
    in_size.setDefault(view.default_size);

    VertexOutput output;
    output.center = in_pos.getPosition(scene.extent);
    output.color  = in_color.getColor();
    output.size   = in_size.getSize();
    return output;
}

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain<M:IMarker>(point VertexOutput input[1], inout TriangleStream<GeometryOutput> stream)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);
    float4 center = mul(input[0].center, view_proj);
    //float4 center = input[0].center;

    GeometryOutput output;
    output.size = SQRT_2*input[0].size + 2 * (v_linewidth + 1.5*u_antialias);

    M marker;
    MarkerOffsets offsets = marker.getVertexOffsets(view_proj, output.size, scene.resolution);
    float4 dx = offsets.dx;
    float4 dy = offsets.dy;

    float4 vertex_offsets[4] = {(-dx-dy), (+dx-dy), (-dx+dy), (+dx+dy)};
    float2 texcoords[4] = {{0,0}, {1,0}, {0,1}, {1,1}};

    output.color = input[0].color;
    output.ele_size = input[0].size;
    //output.normal = normal.xyz;

    // Add point coordinates
    for (int i = 0; i < 4; ++i)
    {
        output.pos = center + vertex_offsets[i];
        output.uv = texcoords[i];
        stream.Append(output);
    }
}

[shader("fragment")]
FragmentOutput fragmentMain<S:IShape2D>(GeometryOutput input)
{
    FragmentOutput out;

    float2 p = 2 * input.uv - 1;
    // if (dot(p, p) > 1) discard;
    // return element.color;
    //float2 p = input.uv - float2(0.5);
    //p = float2(v_rot.x*p.x - v_rot.y*p.y, v_rot.y*p.x + v_rot.x*p.y);

    S shape;
    float distance = shape.apply(p * input.size, input.ele_size);

    Filled filled;
    out.color = filled.apply(distance, v_linewidth, u_antialias, input.color, scene.bg_color);
    //float4x4 view_proj = mul(mvp.view, mvp.proj);
    //float4 clip_pos = mul(view_proj, float4(1, 1, 1, 1));
    //out.depth = clip_pos.z / clip_pos.w;
    out.depth = 1 - out.color.w;
    return out;
}
