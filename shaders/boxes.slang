import uniforms;

import position; // IPosition
import color;    // IColor
import size;     // ISize

struct VertexData
{
    float4 position : POSITION;
    float4 color    : COLOR;
};

[shader("vertex")]
VertexData vertexMain<P : IPosition, C : IColor, S : ISize>(
    P in_pos: POSITION, C in_color: COLOR, S in_size: PSIZE)
{
    VertexData output;
    output.position = in_pos.getPosition(scene.extent);
    output.color = in_color.getColor();

    return output;
}

struct BoxData
{
    float4 pos    : SV_Position;
    float2 uv     : TEXCOORD;
    float4 color  : COLOR;
    float3 normal : NORMAL;
};

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain2D(line VertexData input[2], inout TriangleStream<BoxData> stream)
{
    BoxData output;
    output.color = input[0].color;

    float4 p = input[0].position;
    float2 p0 = p.xy;
    float2 p1 = input[1].position.xy;

    // TODO: Perpective correction
    float2 vertices[4] = { p0, { p1.x, p0.y }, { p0.x, p1.y }, p1 };
    float2 texcoords[4] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };

    for (int i = 0; i < 4; ++i)
    {
        p.xy = vertices[i];
        output.pos = mul(p, mvp.proj);
        output.uv = texcoords[i];
        stream.Append(output);
    }
}

[shader("fragment")]
float4 fragmentMain(BoxData input) : SV_Target
{
    return input.color;

}
