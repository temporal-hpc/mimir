import uniforms;

import position; // IPosition
import color; // IColor
import size; // ISize

struct VertexOutput
{
    float4 position : SV_Position;
    float4 color    : COLOR;
    float4 pos      : POSITION;
};

[shader("vertex")]
VertexOutput vertexMain<P:IPosition, C:IColor, S:ISize>(P in_pos: POSITION, C in_color: COLOR, S in_size: PSIZE)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);

    VertexOutput output;
    float4 pos = in_pos.getPosition(scene.extent);
    output.position = mul(pos, view_proj);
    output.color    = in_color.getColor();

    pos = ((pos + 1) / 2);
    pos.x = pos.x * scene.extent.x;
    pos.y = pos.y * scene.extent.y;
    pos.z = pos.z * scene.extent.z;
    output.pos = pos;

    return output;
}

[shader("geometry")]
[maxvertexcount(6)]
void geometryMain(triangle VertexOutput input[3], inout LineStream<VertexOutput> stream)
{
    VertexOutput output;
    for (int i = 0; i < 3; ++i)
    {
        VertexOutput p1 = input[i];
        VertexOutput p2 = input[(i+1)%3];
        if (distance(p1.pos, p2.pos) <= scene.extent.x / 2.0)
        {
            output.color = p1.color;
            output.position = p1.position;
            stream.Append(output);
            output.color = p2.color;
            output.position = p2.position;
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
