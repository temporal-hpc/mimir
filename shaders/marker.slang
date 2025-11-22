import uniforms;

import position;  // IPosition
import color; // IColor
import size; // ISize
import styles; // IStyle
import shapes; // IShape
import rotation; // IRotation

static const float BOX_CORRECTION = 2.5;

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
    float depth  : SV_Depth;
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
    float2 mapping[4] = { { -1, -1 }, { -1, +1 }, { +1, -1 }, { +1, +1 } };
    G geometry;

    GeometryOutput output;
    output.size = sz;
    output.color = input[0].color;
    output.rotation = input[0].rotation;
    // Add point coordinates
    for (int i = 0; i < 4; ++i)
    {
        float4 offset = float4(mapping[i], 0, 0) * sz;
        float4 vertex = pivot_view + geometry.getVertex(offset) * BOX_CORRECTION;
        output.pos = mul(vertex, mvp.proj);
        output.uv = mapping[i] + float2(1.0);
        output.pivot = pivot_view.xyz;
        output.mapping = mapping[i] * BOX_CORRECTION;
        stream.Append(output);
    }
}

[shader("fragment")]
FragmentOutput fragmentMain<S: IStyle, M: MarkerShape>(GeometryOutput input)
{
    float4 ambient_intensity = { 0.2, 0.2, 0.2, 1.0 };
    float light_attenuation = 1.0 / (25.0 * 25.0);

    float4 diffuse_color = view.default_color;
    float4 specular_color = float4(scene.specular_color, 1);
    float specular_shininess = 0.3;

    float3 camera_space_sphere_pos = input.pivot;
    float sphere_radius = input.size;
    float3 camera_space_plane_pos = float3(input.mapping * sphere_radius, 0.0) + camera_space_sphere_pos;
    float3 ray_dir = normalize(camera_space_plane_pos);

    float B = 2.0 * dot(ray_dir, -camera_space_sphere_pos);
    float C = dot(camera_space_sphere_pos, camera_space_sphere_pos) - (sphere_radius * sphere_radius);

    float det = (B * B) - (4 * C);
    if (det < 0.0) { discard; }

    float sqrt_det = sqrt(det);
    float pos_t = (-B + sqrt_det) / 2.0;
    float neg_t = (-B - sqrt_det) / 2.0;

    float intersect_t = min(pos_t, neg_t);
    float3 camera_space_pos = ray_dir * intersect_t;
    float3 camera_space_normal = normalize(camera_space_pos - camera_space_sphere_pos);

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

    float4 clip_pos = mul(float4(camera_space_pos, 1.0), mvp.proj);
    float ndc_depth = clip_pos.z / clip_pos.w;

    FragmentOutput out;
    out.color = lighting;
    out.depth = ndc_depth;
    return out;
}
