// Path tracing for the particles-kmodal-3d scene (LightModel::PathTracing).
// Phase 2: iterative path tracer in the raygen shader — primary ray, a shadow ray to the sun
// for direct light, and cosine-weighted diffuse bounces for indirect light (global
// illumination) up to pc.bounces. Closest-hit returns hit geometry; miss returns the
// environment/background color. spp samples per pixel per frame are averaged (the scene
// animates every frame, so this is antialiasing + noise reduction, not accumulation).

// Per-particle AABB (VkAabbPositionsKHR: 6 tightly packed floats). The intersection shader reads it
// by primitive index to recover the sphere centre + radius (centre = (min+max)/2, r = (max-min)/2).
struct Aabb
{
    float minx, miny, minz;
    float maxx, maxy, maxz;
};

// Hit attributes produced by the sphere intersection shader and consumed by the closest-hit shader.
// The single TLAS instance is identity, so object space == world space and the normal is already
// world-space (no normal-matrix transform needed).
struct SphereHit
{
    float3 normal;
};

// Must match mimir::RtPushConstants (raytracing.hpp). Camera is an explicit world-space basis
// so the raygen is a plain pinhole (sidesteps mimir's pre-transposed matrix convention). 128
// bytes (with the trailing AABB pointer), at the 128-byte guaranteed push-constant limit.
struct PushConstants
{
    float4 cam_pos;     // xyz camera world position; w = BLAS chunk size (primitives) as uint bits
    float4 cam_right;   // xyz world-space camera basis; w = light_color R (sun tint/intensity)
    float4 cam_up;      // xyz ...                       w = light_color G
    float4 cam_forward; // xyz ...                       w = light_color B
    float4 sun_dir;     // xyz world-space direction TO the sun (normalized); w unused
    float4 sky_color;   // xyz environment/background color, w = intensity
    float tan_half_fov;
    float aspect;
    uint frame_index;
    uint spp;
    uint bounces;
    uint accum_frame;   // temporal accumulation: # of frames already averaged (0 = first/reset)
    // Procedural-sphere geometry: the per-particle AABB buffer as a buffer-device-address pointer
    // (physical storage buffer). The intersection shader reads AABB[PrimitiveIndex] to recover the
    // sphere centre + radius. A pointer (not a descriptor) so it isn't capped by maxStorageBufferRange
    // at large N. Lands at offset 120, keeping the block at exactly the 128-byte guaranteed limit.
    Aabb* aabbs;
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
// Sun radiance per unit of ViewerOptions::light_color (delivered in the camera-basis w lanes),
// so the same light knob drives the raster Phong modes and the path tracer. At the library
// default light_color (0.5 grey) this reproduces the historical fixed sun radiance of 3.0.
static const float SUN_RADIANCE_PER_UNIT = 6.0;

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
    // Sun radiance = light_color (basis w lanes, see PushConstants) x the per-unit scale.
    float3 sun_radiance = SUN_RADIANCE_PER_UNIT
        * float3(pc.cam_right.w, pc.cam_up.w, pc.cam_forward.w);
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
            radiance += throughput * payload.albedo * ndotl * sun_radiance;
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

    // Display. A primary ray that missed all geometry shows the backdrop at its display-referred
    // value -- exactly the raster clear color, so --background 1.0 is truly white -- instead of the
    // Reinhard-tonemapped sky (which would compress 1.0 -> ~0.73 grey). Display-only: bounce rays
    // that escape to the sky still deposit HDR environment light on surfaces (see tracePath), so the
    // light transport is unchanged. g_depth holds the miss sentinel (1e4) for a primary miss.
    float3 disp;
    if (g_depth >= 1e4)
    {
        disp = saturate(pc.sky_color.xyz * pc.sky_color.w);
    }
    else
    {
        disp = mean / (1.0 + mean);                    // Reinhard tone map (compresses the bright sun)
        disp = pow(saturate(disp), float3(1.0 / 2.2)); // gamma
    }
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

// Ray-sphere intersection for the procedural AABB geometry. Each primitive is a sphere whose
// centre + radius are recovered from its AABB (pc.aabbs[PrimitiveIndex]). Reports the nearest hit
// within the ray interval and hands the world-space normal to the closest-hit shader. The single
// TLAS instance is identity, so object space == world space here.
[shader("intersection")]
void sphereIntersect()
{
    // Particles above maxPrimitiveCount are split across several BLASes (one TLAS instance each), so
    // PrimitiveIndex() is LOCAL to this instance's BLAS. Recover the global particle index into the
    // shared AABB buffer: instance c (InstanceID) covers globals [c*chunk, c*chunk+chunk). chunk is
    // packed into cam_pos.w; for a single BLAS InstanceID()==0 so this reduces to PrimitiveIndex().
    // The multiply itself must happen in 64-bit: chunk (blas_chunk_prims) can be up to ~5.3e8, and
    // for InstanceID() >= ~8 the 32-bit product InstanceID()*chunk overflows and wraps, recovering
    // the wrong global index (garbage sphere) for every particle past the 2^32 boundary.
    uint chunk = asuint(pc.cam_pos.w);
    uint64_t gid = uint64_t(InstanceID()) * uint64_t(chunk) + uint64_t(PrimitiveIndex());
    // Explicit 64-bit address + cast, NOT pointer indexing: OpPtrAccessChain offsets are truncated
    // to 32 bits by the NVIDIA compiler even with a 64-bit index, wrapping reads past 4 GiB into
    // the AABB buffer (~179M particles) onto garbage boxes (see pathtrace_aabbs.slang).
    Aabb box = *(Aabb*)(uint64_t(pc.aabbs) + gid * 24ull);
    float3 mn = float3(box.minx, box.miny, box.minz);
    float3 mx = float3(box.maxx, box.maxy, box.maxz);
    float3 center = 0.5 * (mn + mx);
    float  radius = 0.5 * (mx.x - mn.x);

    float3 o = ObjectRayOrigin() - center;
    float3 d = ObjectRayDirection(); // world dir is normalized; identity instance => object dir too
    float b = dot(o, d);
    float c = dot(o, o) - radius * radius;
    float disc = b * b - c;          // dot(d,d) == 1
    if (disc < 0.0) { return; }
    float sq = sqrt(disc);

    float t = -b - sq;               // near root
    if (t < RayTMin()) { t = -b + sq; } // inside the sphere: take the far root
    if (t < RayTMin() || t > RayTCurrent()) { return; }

    SphereHit hit;
    hit.normal = normalize((ObjectRayOrigin() + t * d) - center);
    ReportHit(t, 0u, hit);
}

[shader("closesthit")]
void closestHitMain(inout Payload payload, in SphereHit attr)
{
    // Procedural sphere: the intersection shader already produced the world-space normal (the single
    // TLAS instance is identity, so no normal-matrix transform is needed).
    payload.normal   = attr.normal;
    payload.position = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
    // Surface parameters come from this hit's material (its SBT hit record).
    payload.albedo   = material.albedo;
    payload.emission = material.albedo * material.emission;
    payload.hit_t    = RayTCurrent();
}
