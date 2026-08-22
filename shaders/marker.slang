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
    float4 pivot = mul(mul(input[0].center, mvp.model), mvp.view);
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
        float4 vertex = pivot + geometry.getVertex(offset) * BOX_CORRECTION;
        output.pos = mul(vertex, mvp.proj);
        output.uv = mapping[i] + float2(1.0);
        output.pivot = pivot.xyz;
        output.mapping = mapping[i] * BOX_CORRECTION;
        stream.Append(output);
    }
}

float sphIntersect(float3 ro, float3 rd, float4 sph)
{
    float3 oc = ro - sph.xyz;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sph.w * sph.w;
    float h = b * b - c;
    float rh = (h < 0) ? -1.0 : -b - sqrt(h);
    clip(rh);
    return rh;
}

float sphIntersect2(float3 ro, float3 rd, float4 sph)
{
    float B = 2.0 * dot(rd, -sph.xyz);
    float C = dot(sph.xyz, sph.xyz) - (sph.w * sph.w);
    float det = (B * B) - (4 * C);
    if (det < 0.0) { discard; }
    float sqrt_det = sqrt(det);
    float pos_t = (-B + sqrt_det) / 2.0;
    float neg_t = (-B - sqrt_det) / 2.0;
    return min(pos_t, neg_t);
}

// Blinn-Phong. Robust (no acos, so no NaN speckle at highlights near dot==1) and honours the
// scene's specular_power / ambient_strength (previously hardcoded 0.3/0.2 and ignored). `ld` is the
// unit light direction in view space (toward the light); `li` is the light colour; pos/normal are
// view space. `base` is the surface's own diffuse color: it comes from the marker's COLOR attribute
// when the view binds one (one color per primitive) and falls back to view.default_color otherwise
// -- ColorDefault::getColor() returns exactly that, so an unbound Color attribute is unchanged.
// Returns `base` unchanged under Shading::Unlit (view.shading == 0).
// Kept in sync with shaders/marker_mesh.slang's computeLighting.
float4 computeLighting(float3 pos, float3 normal, float3 ld, float4 li, float4 base) {
    // Shading::Unlit: the marker keeps its own color, flat. The ray-sphere intersection and the
    // depth write above still run, so an unlit impostor is a round, depth-correct sphere -- just
    // not lit. Same guard (and the same view.shading uniform) as voxel.slang.
    if (view.shading < 0.5) { return base; }

    float3 N = normalize(normal);
    float3 V = normalize(-pos);        // camera sits at the origin in view space
    float3 H = normalize(ld + V);      // Blinn-Phong half vector

    float n_dot_l = saturate(dot(N, ld));
    float n_dot_h = saturate(dot(N, H));
    // No specular where the surface faces away from the light.
    float spec = (n_dot_l > 0.0) ? pow(n_dot_h, scene.specular_power) : 0.0;

    float4 diffuse_color  = base;
    float4 specular_color = float4(scene.specular_color, 1.0);

    float4 lighting = diffuse_color * scene.ambient_strength; // ambient
    lighting += diffuse_color * li * n_dot_l;                 // diffuse
    lighting += specular_color * li * spec;                   // specular
    // Opacity is the marker's own alpha, not a lighting term: the sum above (ambient + diffuse*n_dot_l
    // + specular) collapses toward ~ambient_strength on the unlit hemisphere (n_dot_l clamped to 0),
    // and Sphere3D markers blend with SRC_ALPHA/ONE_MINUS_SRC_ALPHA for antialiased impostor edges --
    // so forwarding that near-zero alpha made the dark side of every sphere blend ~95% into the
    // background, looking transparent instead of merely unlit.
    lighting.a = diffuse_color.a;
    return lighting;
}

[shader("fragment")]
FragmentOutput fragmentMain<S: IStyle, M: MarkerShape>(GeometryOutput input)
{
    float3 ray_origin = float3(0);
    float3 sphere_pos = input.pivot;
    float sphere_radius = input.size;
    float3 plane_pos = float3(input.mapping * sphere_radius, 0.0) + sphere_pos;
    float3 ray_dir = normalize(plane_pos);
    float ray_hit = sphIntersect(ray_origin, ray_dir, float4(sphere_pos, sphere_radius));

    float3 pos = ray_dir * ray_hit;
    float3 normal = normalize(pos - sphere_pos);

    // World-space directional light -> view space (view only, NOT model: the impostor sphere normals
    // are isotropic view-space geometry, so a per-view model rotation must never rotate the light).
    // Normalized so diffuse magnitude stays correct even if the view matrix ever carries scale.
    float3 light_dir = normalize(mul(float4(scene.light_pos, 0), mvp.view).xyz);
    float4 light_intensity = float4(scene.light_color, 1);
    // Per-primitive diffuse color: input.color is the vertex COLOR attribute carried through the
    // geometry stage (view.default_color when the view binds no Color attribute).
    float4 lighting = computeLighting(pos, normal, light_dir, light_intensity, input.color);

    float4 clip_pos = mul(float4(pos, 1.0), mvp.proj);

    FragmentOutput out;
    out.color = lighting;
    out.depth = clip_pos.z / clip_pos.w;
    return out;
}