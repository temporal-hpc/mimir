// Fullscreen composite: samples the ray-traced storage image into the raster color
// attachment so the existing render pass + ImGui HUD draw on top unchanged. Draws a single
// oversized triangle (3 vertices, no vertex buffer) that covers the viewport.

struct VertexData
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD;
};

[shader("vertex")]
VertexData vertexMain(uint vid : SV_VertexID)
{
    VertexData output;
    // Fullscreen triangle: uv in {0,2} -> clip in {-1,3}, covering [-1,1] after clipping.
    output.uv  = float2((vid << 1) & 2, vid & 2);
    output.pos = float4(output.uv * 2.0 - 1.0, 0.0, 1.0);
    return output;
}

[[vk::binding(0, 0)]] Sampler2D<float4> traced;

[shader("fragment")]
float4 fragmentMain(VertexData input) : SV_Target
{
    return float4(traced.Sample(input.uv).rgb, 1.0);
}
