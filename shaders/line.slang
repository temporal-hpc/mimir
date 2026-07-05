import uniforms;

import position; // IPosition
import color; // IColor
import size; // ISize
import texcoord; // ITexcoord (optional screen-space shift for billboarded overlays)

struct VertexOutput
{
    float4 pos   : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vertexMain<P:IPosition, C:IColor, S:ISize, T:ITexcoord>(
    P in_pos: POSITION, C in_color: COLOR, S in_size: PSIZE, T in_shift: TEXCOORD)
{
    VertexOutput output;
    float4 pos  = in_pos.getPosition();
    float4 clip = mul(pos, mvp.all);
    // Optional screen-space shift (TexcoordDefault = 0 for plain world-space lines), applied
    // after projection so shifted geometry always faces the camera (billboard). Same math as
    // datoviz's transform_shift (2*shift/viewport, not scaled by w), so gizmos built this way
    // render identically in both: +x right, +y up, on-screen pixels = shift / clip.w.
    float2 shift = in_shift.getTexcoord();
    clip.x += 2.0 * shift.x / scene.resolution.x;
    clip.y -= 2.0 * shift.y / scene.resolution.y; // Vulkan clip y grows downward
    output.pos   = clip;
    output.color = in_color.getColor();
    //output.size  = in_size.getSize();

    return output;
}

// [shader("geometry")]
// [maxvertexcount(4)]
// void geometryMain(line VertexOutput input[2], inout TriangleStream<VertexOutput> stream)
// {
//     float4 dx = mvp.all[0] / 2.0 * input[0].size;
//     float4 dy = mvp.all[1] / 2.0 * input[0].size;
//     float4 offsets[4] = { (-dx - dy), (+dx - dy), (-dx + dy), (+dx + dy) };

//     VertexOutput output;
//     output.color = input[0].color;
//     output.size  = input[0].size;

//     output.pos = input[0].pos + offsets[0];
//     stream.Append(output);
//     output.pos = input[1].pos + offsets[1];
//     stream.Append(output);
//     output.pos = input[0].pos + offsets[2];
//     stream.Append(output);
//     output.pos = input[1].pos + offsets[3];
//     stream.Append(output);
// }

[shader("geometry")]
[maxvertexcount(6)]
void geometryPBCMain(triangle VertexOutput input[3], inout LineStream<VertexOutput> stream)
{
    VertexOutput output;
    for (int i = 0; i < 3; ++i)
    {
        VertexOutput p1 = input[i];
        VertexOutput p2 = input[(i+1)%3];
        if (distance(p1.pos, p2.pos) <= (scene.extent.x * mvp.model[0][0]) / 2.0)
        {
            output.color = p1.color;
            output.pos = p1.pos;
            stream.Append(output);
            output.color = p2.color;
            output.pos = p2.pos;
            stream.Append(output);
            stream.RestartStrip();
        }
    }
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target
{
    return input.color;
}
