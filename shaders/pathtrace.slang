// Path tracing for the particles-kmodal-3d scene (LightModel::PathTracing).
// Phase 1: primary rays against a static icosphere-grid TLAS, lambertian sun + sky miss.
// One shader module, RT-pipeline entry points (raygen / miss / closest-hit) grouped into
// the SBT by the engine. Bounces/shadows arrive in Phase 2.

// Must match mimir::RtPushConstants (raytracing.hpp). Camera is an explicit world-space
// basis (from the LookAt camera-to-world matrix) so the raygen is a plain pinhole; this
// sidesteps mimir's pre-transposed glm/slang matrix convention.
struct PushConstants
{
    float4 cam_pos;     // camera world position
    float4 cam_right;   // world-space camera basis
    float4 cam_up;
    float4 cam_forward;
    float4 sun_dir;     // world-space direction TO the sun (normalized)
    float4 sky_color;   // xyz environment/background color, w = intensity
    float tan_half_fov;
    float aspect;
    uint frame_index;
    uint spp;
    uint bounces;
    uint pad0;
    uint pad1;
    uint pad2;
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> image;

struct Payload
{
    float3 color;
    float hit_t; // < 0 on miss
};

[shader("raygeneration")]
void raygenMain()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims  = DispatchRaysDimensions().xy;

    float2 uv  = (float2(pixel) + 0.5) / float2(dims);
    float2 ndc = uv * 2.0 - 1.0; // [-1,1], y increases downward (Vulkan pixel order)

    // Pinhole primary ray in world space. Screen-up is +cam_up, so negate ndc.y (top row
    // ndc.y=-1 must map to +up). Horizontal spread scales by aspect.
    float3 dir = pc.cam_forward.xyz
               + ndc.x * pc.tan_half_fov * pc.aspect * pc.cam_right.xyz
               + (-ndc.y) * pc.tan_half_fov * pc.cam_up.xyz;

    RayDesc ray;
    ray.Origin    = pc.cam_pos.xyz;
    ray.Direction = normalize(dir);
    ray.TMin      = 1e-3;
    ray.TMax      = 1e4;

    Payload payload;
    payload.color = float3(0.0);
    payload.hit_t = -1.0;
    TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

    // Gamma (tone mapping proper arrives with HDR bounces in Phase 2).
    float3 color = pow(saturate(payload.color), float3(1.0 / 2.2));
    image[pixel] = float4(color, 1.0);
}

[shader("miss")]
void missMain(inout Payload payload)
{
    // Environment/background color, driven by ViewerOptions::background_color (w = intensity).
    // This is both what a missed ray shows as the backdrop AND the scene's ambient fill, so a
    // simulation can pick any background — including black, which lights the scene by sun alone.
    payload.color = pc.sky_color.xyz * pc.sky_color.w;
    payload.hit_t = -1.0;
}

[shader("closesthit")]
void closestHitMain(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // The BLAS vertices lie on a unit sphere centered at the object origin, so the smooth
    // surface normal is just the normalized object-space hit position. Phase 1 instances
    // carry no rotation (uniform scale + translation only), so the object normal equals the
    // world normal; a proper normal-matrix transform lands with cube rotation in Phase 3.
    float3 obj_pos = ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection();
    float3 normal  = normalize(obj_pos);

    float3 albedo  = float3(0.82, 0.82, 0.88);
    float3 sun     = normalize(pc.sun_dir.xyz);
    float ndotl    = max(dot(normal, sun), 0.0);

    // Direct sun + ambient fill from the environment (background_color), so shadowed sides
    // pick up the backdrop's tint. A black background means sun-only lighting.
    float3 ambient = pc.sky_color.xyz * pc.sky_color.w * 0.35;
    payload.color  = albedo * (ndotl + ambient);
    payload.hit_t  = RayTCurrent();
}
