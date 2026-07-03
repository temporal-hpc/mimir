// Path tracing for the particles-kmodal-3d scene (LightModel::PathTracing).
// Phase 2: iterative path tracer in the raygen shader — primary ray, a shadow ray to the sun
// for direct light, and cosine-weighted diffuse bounces for indirect light (global
// illumination) up to pc.bounces. Closest-hit returns hit geometry; miss returns the
// environment/background color. spp samples per pixel per frame are averaged (the scene
// animates every frame, so this is antialiasing + noise reduction, not accumulation).

// Must match mimir::RtPushConstants (raytracing.hpp). Camera is an explicit world-space basis
// so the raygen is a plain pinhole (sidesteps mimir's pre-transposed matrix convention).
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

static const float PI = 3.14159265;
static const float SUN_INTENSITY = 3.0; // radiance scale of the directional sun

struct Payload
{
    float3 color;    // miss: environment color
    float  hit_t;    // >= 0 on hit (distance), < 0 on miss
    float3 normal;   // world-space surface normal at the hit
    float3 position; // world-space hit position
    float3 albedo;   // surface albedo at the hit
};

// PCG hash for cheap per-sample jitter and hemisphere sampling (stateful; advance each draw).
uint pcgNext(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rnd(inout uint state) { return float(pcgNext(state)) / 4294967296.0; }

// Cosine-weighted hemisphere sample around n (for the lambertian bounce; pdf = cos/PI, so the
// path throughput just multiplies by the albedo).
float3 cosineSampleHemisphere(float3 n, inout uint seed)
{
    float u1 = rnd(seed);
    float u2 = rnd(seed);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float3 local = float3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));

    float3 up = abs(n.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 t = normalize(cross(up, n));
    float3 b = cross(n, t);
    return normalize(t * local.x + b * local.y + n * local.z);
}

// True if the sun is occluded from p along L (shadow ray). Reuses the environment miss shader:
// on a hit, closest-hit is skipped and hit_t stays >= 0 (occluded); on a miss, the miss shader
// sets hit_t < 0 (visible).
bool sunOccluded(float3 p, float3 L)
{
    RayDesc s;
    s.Origin = p;
    s.Direction = L;
    s.TMin = 1e-3;
    s.TMax = 1e4;
    Payload sp;
    sp.hit_t = 0.0; // assume occluded until the miss shader proves otherwise
    TraceRay(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF, 0, 0, 0, s, sp);
    return sp.hit_t >= 0.0;
}

// Trace one full path and return its radiance.
float3 tracePath(float3 origin, float3 dir, inout uint seed)
{
    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);
    float3 sun = normalize(pc.sun_dir.xyz);

    // bounces + 1 segments: the primary ray plus pc.bounces indirect bounces.
    uint segments = pc.bounces + 1u;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 1e-3;
    ray.TMax = 1e4;

    for (uint b = 0; b < segments; ++b)
    {
        Payload payload;
        payload.hit_t = -1.0;
        TraceRay(tlas, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, payload);

        if (payload.hit_t < 0.0)
        {
            // Missed geometry: pick up the environment (also the indirect sky fill).
            radiance += throughput * payload.color;
            break;
        }

        // Direct sun with a shadow ray.
        float ndotl = max(dot(payload.normal, sun), 0.0);
        if (ndotl > 0.0 && !sunOccluded(payload.position + payload.normal * 1e-3, sun))
        {
            radiance += throughput * payload.albedo * ndotl * SUN_INTENSITY;
        }

        // Diffuse bounce (lambertian): throughput *= albedo, new cosine-weighted direction.
        throughput *= payload.albedo;
        ray.Origin = payload.position + payload.normal * 1e-3;
        ray.Direction = cosineSampleHemisphere(payload.normal, seed);
    }
    return radiance;
}

[shader("raygeneration")]
void raygenMain()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 dims  = DispatchRaysDimensions().xy;

    // Per-pixel, per-frame RNG so spp>1 jitters and paths differ each frame.
    uint seed = (pixel.x * 1973u + pixel.y * 9277u + pc.frame_index * 26699u) | 1u;
    uint spp  = max(pc.spp, 1u);

    float3 accum = float3(0.0);
    for (uint s = 0; s < spp; ++s)
    {
        // Jitter the pixel by [-0.5,0.5] for antialiasing (no jitter when spp == 1).
        float2 jitter = (spp > 1u) ? float2(rnd(seed), rnd(seed)) - 0.5 : float2(0.0);
        float2 uv  = (float2(pixel) + 0.5 + jitter) / float2(dims);
        float2 ndc = uv * 2.0 - 1.0; // [-1,1], y increases downward (Vulkan pixel order)

        // Pinhole primary ray in world space. Screen-up is +cam_up, so negate ndc.y.
        float3 dir = normalize(pc.cam_forward.xyz
                   + ndc.x * pc.tan_half_fov * pc.aspect * pc.cam_right.xyz
                   + (-ndc.y) * pc.tan_half_fov * pc.cam_up.xyz);
        accum += tracePath(pc.cam_pos.xyz, dir, seed);
    }

    // Average, tone map (Reinhard) so the bright sun compresses into range, then gamma.
    float3 color = accum / float(spp);
    color = color / (1.0 + color);
    color = pow(saturate(color), float3(1.0 / 2.2));
    image[pixel] = float4(color, 1.0);
}

[shader("miss")]
void missMain(inout Payload payload)
{
    // Environment/background color, driven by ViewerOptions::background_color (w = intensity).
    // Missed rays (primary or bounce) pick this up as sky/fill; hit_t < 0 also signals a shadow
    // ray that the sun is visible.
    payload.color = pc.sky_color.xyz * pc.sky_color.w;
    payload.hit_t = -1.0;
}

[shader("closesthit")]
void closestHitMain(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    // BLAS vertices lie on a unit sphere centered at the object origin, so the smooth normal is
    // the normalized object-space hit position. Phase 2 instances carry no rotation (uniform
    // scale + translation), so the object normal equals the world normal; a proper normal-matrix
    // transform lands with cube rotation in Phase 3.
    float3 obj_pos = ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection();
    payload.normal   = normalize(obj_pos);
    payload.position = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    payload.albedo   = float3(0.82, 0.82, 0.88);
    payload.hit_t    = RayTCurrent();
}
