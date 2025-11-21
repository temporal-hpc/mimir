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
    float4 pos : SV_Position;
    float2 uv    : TEXCOORD;
    float4 color : COLOR;
    float size   : PSIZE;
    float2 rotation;
    float3 pivot;
    float2 mapping;
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
    float2 mapping[4] = { { -1, -1 }, { -1, +1 }, { +1, -1 }, { +1, +1 } };
    float2 uv[4] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
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
        output.uv = uv[i];
        output.pivot = pivot_view.xyz;
        output.mapping = mapping[i];
        stream.Append(output);
    }
}

[shader("fragment")]
FragmentOutput fragmentMain<S: IStyle, M: MarkerShape>(GeometryOutput input)
{
    // float2 p = input.uv - float2(0.5);
    // float2 r = input.rotation;
    // p = float2(r.x * p.x - r.y * p.y, r.y * p.x + r.x * p.y);
    // M shape;
    // float distance = shape.apply(p * getMarkerSizePx(input.size), input.size);
    // S style;
    // float4 color = style.apply(distance, view.linewidth, view.antialias, input.color, scene.background_color);

    float4 ambient_intensity = { 0.2, 0.2, 0.2, 1.0 };
    float light_attenuation = 1.0 / (25.0 * 25.0);

    float4 diffuse_color = { 0.4, 0.4, 0.4, 1.0 };
    float4 specular_color = float4(scene.specular_color, 1);
    float specular_shininess = 0.3;

    float lensqr = dot(input.mapping, input.mapping);
    if (lensqr > 1.0) { discard; }
    float3 camera_space_normal = float3(input.mapping, sqrt(1.0 - lensqr));
    float3 camera_space_pos = (camera_space_normal * input.size) + input.pivot;

    float4 light_pos = mul(float4(scene.light_pos, 0), mvp.model);
    float3 light_dir = mul(light_pos, mvp.view).xyz;
    float4 light_intensity = float4(scene.light_color, 1);

    float3 surface_normal = normalize(camera_space_normal);
    float cos_ang_incidence = dot(surface_normal, light_dir);
    cos_ang_incidence = cos_ang_incidence < 0.0001 ? 0.0 : cos_ang_incidence;

    float3 view_direction = normalize(-camera_space_pos);
    float3 half_angle = normalize(light_dir + view_direction);
    float angle_normal_half = acos(dot(half_angle, surface_normal));
    float exponent = angle_normal_half / specular_shininess;
    exponent = -(exponent * exponent);
    float gaussian_term = exp(exponent);

    float4 lighting = diffuse_color * ambient_intensity;
    lighting += diffuse_color * light_intensity * cos_ang_incidence;
    lighting += specular_color * light_intensity * gaussian_term;

    FragmentOutput out;
    out.color = lighting;
    return out;
}

float getMarkerSizePx(float size)
{
    return SQRT_2 * size + 2 * (view.linewidth + 1.5 * view.antialias);
}