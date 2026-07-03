// À-trous edge-avoiding wavelet denoiser (Dammertz et al. 2010) for the path-traced result.
// This is the "runs on any GPU" fallback denoiser (a Vulkan compute pass); the OptiX AI denoiser
// takes over when tensor cores are available. Each invocation applies a 5x5 wavelet with an
// increasing step_width across passes, so a few passes reach a wide spatial support cheaply. The
// filter is edge-aware: neighbor contributions are weighted down across discontinuities in colour,
// surface normal, and depth (the first-hit G-buffer), so silhouettes and shading edges survive.

struct PushConstants
{
    int2 dims;       // image size in pixels
    int  step_width; // à-trous hole size for this pass (1, 2, 4, ...)
    float c_phi;     // colour edge-stopping falloff (larger = more blurring across colour)
    float n_phi;     // normal edge-stopping falloff
    float p_phi;     // depth edge-stopping falloff
    uint tonemap;    // 1 = tonemap+gamma the output (final pass to the display image), 0 = keep HDR
};
[[vk::push_constant]] PushConstants pc;

[[vk::binding(0, 0)]] RWTexture2D<float4> in_color;  // noisy / partially-filtered HDR radiance
[[vk::binding(1, 0)]] RWTexture2D<float4> gbuffer;   // xyz = world normal, w = eye-space depth
[[vk::binding(2, 0)]] RWTexture2D<float4> out_color; // filtered result (HDR, or tonemapped if final)

// Separable B3 spline kernel {1/16, 1/4, 3/8, 1/4, 1/16}, applied as an outer product over 5x5.
static const float kWeight[5] = { 0.0625, 0.25, 0.375, 0.25, 0.0625 };

[shader("compute")]
[numthreads(8, 8, 1)]
void atrousMain(uint3 tid : SV_DispatchThreadID)
{
    int2 p = int2(tid.xy);
    if (p.x >= pc.dims.x || p.y >= pc.dims.y) { return; }

    float3 c_center = in_color[p].xyz;
    float4 g_center = gbuffer[p];
    float3 n_center = g_center.xyz;
    float  d_center = g_center.w;

    float3 sum = float3(0.0);
    float  wsum = 0.0;
    for (int dy = -2; dy <= 2; ++dy)
    {
        for (int dx = -2; dx <= 2; ++dx)
        {
            int2 q = clamp(p + int2(dx, dy) * pc.step_width, int2(0, 0), pc.dims - 1);
            float3 c = in_color[q].xyz;
            float4 g = gbuffer[q];

            // Edge-stopping weights: colour, normal, and depth similarity (Gaussian in each).
            float3 dc = c - c_center;
            float w_c = exp(-dot(dc, dc) / max(pc.c_phi, 1e-6));
            float3 dn = g.xyz - n_center;
            float w_n = exp(-dot(dn, dn) / max(pc.n_phi, 1e-6));
            float  dd = g.w - d_center;
            float w_p = exp(-(dd * dd) / max(pc.p_phi, 1e-6));

            float w = kWeight[dx + 2] * kWeight[dy + 2] * w_c * w_n * w_p;
            sum  += c * w;
            wsum += w;
        }
    }

    float3 result = (wsum > 0.0) ? sum / wsum : c_center;
    if (pc.tonemap != 0u)
    {
        result = result / (1.0 + result);          // Reinhard, matching the non-denoised path
        result = pow(saturate(result), float3(1.0 / 2.2));
    }
    out_color[p] = float4(result, 1.0);
}
