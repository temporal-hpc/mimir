import uniforms;

import position; // IPosition
import color;    // IColor
import size;     // ISize

struct VertexData
{
    float4 position : SV_Position;
    float4 color    : COLOR;
};

[shader("vertex")]
VertexData vertexMain<P : IPosition, C : IColor, S : ISize>(
    P in_pos: POSITION, C in_color: COLOR, S in_size: PSIZE)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);

    VertexData output;
    float4 pos = in_pos.getPosition(scene.extent);
    pos.z -= 0.001f;
    output.position = pos; //mul(pos, view_proj);
    output.color = in_color.getColor();

    return output;
}

struct BoxData
{
    float4 pos    : SV_Position;
    float4 color  : COLOR;
};

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain2D(line VertexData input[2], inout TriangleStream<BoxData> stream)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);

    BoxData output;

    float4 p0 = input[0].position;
    float4 p1 = input[1].position;
    output.pos = p0;
    output.color = { p0.x, p1.y, 1.f, 1.f };

    float2 vertices[4] = { { p0.x, p1.y }, p1.xy, p0.xy, { p1.x, p0.y } };
    // float2 texcoords[4] = { { 0, 0 }, { 1, 0 }, { 0, 1 }, { 1, 1 } };
    float4 colors[4] = {
        { 0.f, 0.f, 1.f, 1.f },
        { 0.f, 1.f, 0.f, 1.f },
        { 1.f, 0.f, 0.f, 1.f },
        { 0.f, 0.f, 1.f, 1.f }
    };

    // output.pos = mul(p0, view_proj);
    // output.color = colors[2];
    // stream.Append(output);
    // output.pos = mul(p1, view_proj);
    // output.color = colors[1];
    // stream.Append(output);

    for (int i = 0; i < 4; ++i)
    {
        float4 p = p0;
        p.xy = vertices[i];
        output.pos = mul(p, view_proj);
        //output.uv = texcoords[i];
        output.color = colors[i];
        stream.Append(output);
    }
}

[shader("fragment")]
float4 fragmentMain(BoxData input) : SV_Target
{
    return input.color;

}