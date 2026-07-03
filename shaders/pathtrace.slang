// Path tracing for the particles-kmodal-3d scene (LightModel::PathTracing).
// Phase 2: iterative path tracer in the raygen shader — primary ray, a shadow ray to the sun
// for direct light, and cosine-weighted diffuse bounces for indirect light (global
// illumination) up to pc.bounces. Closest-hit returns hit geometry; miss returns the
// environment/background color. spp samples per pixel per frame are averaged (the scene
// animates every frame, so this is antialiasing + noise reduction, not accumulation).

// Must match mimir::RtPushConstants (raytracing.hpp). Camera is an explicit world-space basis
// so the raygen is a plain pinhole (sidesteps mimir's pre-transposed matrix convention). 120
// bytes, within the 128-byte guaranteed push-constant limit.
struct PushConstants
{
    float4 cam_pos;     // xyz camera world position; w unused
    float4 cam_right;   // xyz world-space camera basis; w unused
    float4 cam_up;      // xyz ...
    float4 cam_forward; // xyz ...
    float4 sun_dir;     // xyz world-space direction TO the sun (normalized); w unused
    float4 sky_color;   // xyz environment/background color, w = intensity
    float tan_half_fov;
    float aspect;
    uint frame_index;
    uint spp;
    uint bounces;
    uint accum_frame;   // temporal accumulation: # of frames already averaged (0 = first/reset)
};
[[vk::push_constant]] PushConstants pc;

// Per-material shader-record data, laid out in each SBT hit record after the group handle. The
// instance's shaderBindingTableRecordOffset selects which record (hence which material) a hit
// uses, so different instances can carry different surface parameters from one closest-hit shader.
// Must match mimir::MaterialData (raytracing.hpp): float3 albedo + float emission (16 bytes).
struct MaterialData
{
    float3 albedo;   // diffuse reflectance
    float  emission; // 0 = plain lambertian; >0 emits albedo*emission as radiance (area light)
};
[[vk::shader_record]] ConstantBuffer<MaterialData> material;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] RWTexture2D<float4> image; // display output (tonemapped), composited to screen
// Persistent HDR running-mean accumulator (shared across frames in flight, RGBA32F). Holds the
// pre-tonemap radiance averaged over accum_frame+1 frames; reset by restarting from accum_frame 0.
[[vk::binding(2, 0)]] RWTexture2D<float4> accum;
// First-hit G-buffer guide for the denoiser (RGBA16F): xyz = world-space normal, w = hit distance
// (linear depth from the eye). Miss writes a zero normal and a large depth so the filter treats
// background as a flat region. Consumed by the a-trous edge-avoiding wavelet denoiser.
[[vk::binding(3, 0)]] RWTexture2D<float4> gbuffer;

static const float PI = 3.14159265;
static const float SUN_INTENSITY = 3.0; // radiance scale of the directional sun

struct Payload
{
    float3 color;    // miss: environment color
    float  hit_t;    // >= 0 on hit (distance), < 0 on miss
    float3 normal;   // world-space surface normal at the hit
    float3 position; // world-space hit position
    float3 albedo;   // surface albedo at the hit (from the hit material's SBT record)
    float3 emission; // emitted radiance at the hit (albedo*emission; 0 for non-emissive materials)
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

// Trace one full path and return its radiance. Also reports the primary-hit guide (world normal
// and eye-space depth) for the denoiser: g_normal is the first-hit surface normal (0 on a miss),
// g_depth is the primary hit distance (a large sentinel on a miss).
float3 tracePath(float3 origin, float3 dir, inout uint seed,
    out float3 g_normal, out float g_depth)
{
    float3 radiance   = float3(0.0);
    float3 throughput = float3(1.0);
    float3 sun = normalize(pc.sun_dir.xyz);
    g_normal = float3(0.0);
    g_depth  = 1e4; // miss sentinel (far)

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

        // Capture the primary-hit guide (denoiser edge-stopping) on the first segment.
        if (b == 0u) { g_normal = payload.normal; g_depth = payload.hit_t; }

        // Emissive materials add their own radiance regardless of external lighting (area lights).
        radiance += throughput * payload.emission;

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

    float3 sample_sum = float3(0.0);
    float3 g_normal = float3(0.0);
    float  g_depth  = 1e4;
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
        float3 sn; float sd;
        sample_sum += tracePath(pc.cam_pos.xyz, dir, seed, sn, sd);
        // Keep the un-jittered (or first) sample's guide; all samples share the same primary hit.
        if (s == 0u) { g_normal = sn; g_depth = sd; }
    }
    // Write the first-hit guide for the denoiser (stable across accumulation for a static view).
    gbuffer[pixel] = float4(g_normal, g_depth);

    // This frame's raw HDR radiance (mean over spp), pre-tonemap.
    float3 color = sample_sum / float(spp);

    // Temporal accumulation: keep a running mean of the HDR radiance across frames. accum_frame
    // is the number of frames already averaged into the buffer (0 on the first frame after a
    // reset, which the host triggers on any camera move or a new simulation iteration). A static
    // view therefore converges (noise falls as 1/sqrt(N)); a changing scene restarts from scratch.
    uint n = pc.accum_frame;
    float3 mean;
    if (n == 0u) { mean = color; }
    else
    {
        float3 prev = accum[pixel].xyz;
        mean = (prev * float(n) + color) / float(n + 1u);
    }
    accum[pixel] = float4(mean, 1.0);

    // Tone map (Reinhard) so the bright sun compresses into range, then gamma, for display.
    float3 disp = mean / (1.0 + mean);
    disp = pow(saturate(disp), float3(1.0 / 2.2));
    image[pixel] = float4(disp, 1.0);
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
    // the normalized object-space hit position.
    float3 obj_pos = ObjectRayOrigin() + RayTCurrent() * ObjectRayDirection();
    float3 obj_n   = normalize(obj_pos);

    // Transform the object-space normal to world space with the normal matrix, i.e. the
    // inverse-transpose of the object->world matrix. Normals transform by inverse-transpose so
    // that non-uniform scale / rotation keeps them perpendicular to the surface (a plain
    // ObjectToWorld would shear them off the surface). WorldToObject3x4() is already the inverse
    // of ObjectToWorld, so mul(obj_n, worldToObject) == transpose(worldToObject) * obj_n gives the
    // inverse-transpose product without an explicit transpose. Renormalize (scale changes length).
    // The current spheres carry only uniform scale + translation, so this reduces to the identity
    // rotation here, but it's the correct general form for rotated / non-uniformly scaled geometry.
    float3x3 world_to_object = (float3x3)WorldToObject3x4();
    payload.normal   = normalize(mul(obj_n, world_to_object));
    payload.position = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    // Surface parameters come from this instance's material (its SBT hit record), so different
    // instances shade differently through the same closest-hit shader.
    payload.albedo   = material.albedo;
    payload.emission = material.albedo * material.emission;
    payload.hit_t    = RayTCurrent();
}
