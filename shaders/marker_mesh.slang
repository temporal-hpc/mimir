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
    float4 color : COLOR; // per-instance diffuse color (view.default_color when none is bound)
};

// Shared vertex body; `base` is the instance's diffuse color (see the two entry points below).
VertexOutput buildVertex(float3 in_local, float3 in_center, float3 in_normal, float4 base)
{
    float4 center_view = mul(mul(float4(in_center, 1.0), mvp.model), mvp.view);
    // Offset the template vertex (already unit for the sphere; +/-1 corner for the cube) by the marker
    // size; the normal is supplied per-vertex (sphere: == position; cube: per-face), not derived. The
    // marker model is identity in this sample, so the view-space normal matches the previous
    // normalize(in_local) for the sphere; the cube gets true flat per-face normals.
    float3 vpos = center_view.xyz + in_local * view.default_size;

    VertexOutput output;
    output.pos     = mul(float4(vpos, 1.0), mvp.proj);
    output.vpos    = vpos;
    output.vnormal = normalize(in_normal);
    output.color   = base;
    return output;
}

// Default entry point: the whole cloud shades with the view's single default_color. Bound when the
// view has no Color attribute (see getVertexDescription / getShaderCompileParams in pipeline.cpp,
// which pick the entry point and the matching vertex-input layout together).
[shader("vertex")]
VertexOutput vertexMain(
    [[vk::location(0)]] float3 in_local  : POSITION,   // template vertex, local space (per-vertex)
    [[vk::location(1)]] float3 in_center : TEXCOORD0,  // particle/cell center     (per-instance)
    [[vk::location(2)]] float3 in_normal : NORMAL)     // template vertex normal   (per-vertex)
{
    return buildVertex(in_local, in_center, in_normal, view.default_color);
}

// Per-primitive color entry point: one color per marker, delivered as a THIRD, per-instance vertex
// binding (location 3). Declared float4 whatever the source format is -- Vulkan's vertex-input stage
// converts, and a 3-component format fills the missing alpha with 1 -- so float4/float3/uchar4 color
// buffers all land here without a shader variant each.
[shader("vertex")]
VertexOutput vertexColorMain(
    [[vk::location(0)]] float3 in_local  : POSITION,   // template vertex, local space (per-vertex)
    [[vk::location(1)]] float3 in_center : TEXCOORD0,  // particle/cell center     (per-instance)
    [[vk::location(2)]] float3 in_normal : NORMAL,     // template vertex normal   (per-vertex)
    [[vk::location(3)]] float4 in_color  : COLOR)      // marker color             (per-instance)
{
    return buildVertex(in_local, in_center, in_normal, in_color);
}

// Blinn-Phong. Robust (no acos, so no NaN speckle at highlights near dot==1) and honours the
// scene's specular_power / ambient_strength (previously hardcoded 0.3/0.2 and ignored). `ld` is the
// unit light direction in view space (toward the light); `li` is the light colour; pos/normal are
// view space. `base` is the marker's own diffuse color (its Color attribute, or view.default_color).
// Kept in sync with shaders/marker.slang's computeLighting.
float4 computeLighting(float3 pos, float3 normal, float3 ld, float4 li, float4 base)
{
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
    // Opacity is the marker's own alpha, not a lighting term -- see marker.slang's computeLighting
    // for why (this copy is inert today since SphereMesh disables blending, but keep both in sync).
    lighting.a = diffuse_color.a;
    return lighting;
}

struct FragmentOutput
{
    float4 color : SV_Target;
};

[shader("fragment")]
FragmentOutput fragmentMain(VertexOutput input)
{
    // World-space directional light -> view space (view only, NOT model), same as the impostor path.
    float3 ld = normalize(mul(float4(scene.light_pos, 0), mvp.view).xyz);
    float4 li = float4(scene.light_color, 1);

    FragmentOutput output;
    output.color = computeLighting(input.vpos, normalize(input.vnormal), ld, li, input.color);
    return output;
}
