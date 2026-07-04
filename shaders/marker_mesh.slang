// Instanced triangle-icosphere markers (LightModel::PhongMesh / RenderMode::SphereMesh).
//
// One indexed instance of a shared unit icosphere is drawn per particle. Binding 0 holds the
// template vertex positions (which double as unit normals); binding 1 holds the interop particle
// centers as per-instance data. Unlike the Sphere3D impostor there is no geometry stage, no
// per-fragment ray-sphere intersection and no shader depth write, so early-Z culls overdraw.
//
// The sphere is assembled in view space (center -> view, then offset by the unit direction scaled
// to the marker radius), so lighting uses the same view-space Blinn-Phong convention as the
// impostor path -- including the fly camera's world-to-view and light_pos handling, which arrive
// through the shared mvp/scene uniforms.
import uniforms;

struct VertexOutput
{
    float4 pos     : SV_Position;
    float3 vpos;    // view-space position (for the view direction / specular)
    float3 vnormal; // view-space normal (unit sphere direction)
};

[shader("vertex")]
VertexOutput vertexMain(
    [[vk::location(0)]] float3 in_local  : POSITION,   // template icosphere vertex (per-vertex)
    [[vk::location(1)]] float3 in_center : TEXCOORD0)  // particle center       (per-instance)
{
    float4 center_view = mul(mul(float4(in_center, 1.0), mvp.model), mvp.view);
    float3 n = normalize(in_local);
    float3 vpos = center_view.xyz + n * view.default_size;

    VertexOutput output;
    output.pos     = mul(float4(vpos, 1.0), mvp.proj);
    output.vpos    = vpos;
    output.vnormal = n;
    return output;
}

// Blinn-Phong with a Gaussian specular lobe, matching shaders/marker.slang's computeLighting.
float4 computeLighting(float3 pos, float3 normal, float3 ld, float4 li)
{
    float4 ambient_intensity = { 0.2, 0.2, 0.2, 1.0 };
    float4 diffuse_color  = view.default_color;
    float4 specular_color = float4(scene.specular_color, 1);
    float specular_shininess = 0.3;

    float cos_ang_incidence = dot(normal, ld);
    cos_ang_incidence = cos_ang_incidence < 0.0001 ? 0.0 : cos_ang_incidence;

    float3 view_dir = normalize(-pos);
    float3 half_angle = normalize(ld + view_dir);
    float angle_normal_half = acos(clamp(dot(half_angle, normal), -1.0, 1.0));
    float exponent = angle_normal_half / specular_shininess;
    exponent = -(exponent * exponent);
    float gaussian_term = cos_ang_incidence != 0.0 ? exp(exponent) : 0.0;

    float4 lighting = diffuse_color * ambient_intensity;
    lighting += diffuse_color * li * cos_ang_incidence;
    lighting += specular_color * li * gaussian_term;
    return lighting;
}

struct FragmentOutput
{
    float4 color : SV_Target;
};

[shader("fragment")]
FragmentOutput fragmentMain(VertexOutput input)
{
    // Light direction transformed into view space (direction, w=0), same as the impostor path.
    float3 ld = normalize(mul(mul(float4(scene.light_pos, 0), mvp.model), mvp.view).xyz);
    float4 li = float4(scene.light_color, 1);

    FragmentOutput output;
    output.color = computeLighting(input.vpos, normalize(input.vnormal), ld, li);
    return output;
}
