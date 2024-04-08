#version 460 core

layout (points) in;
layout (triangle_strip, max_vertices=4) out;
out vec2 point_uv;
out float point_size;

void main()
{
    ivec2 resolution = {1920, 1080};
    float SQRT_2 = 1.4142135623730951f;

    point_size = SQRT_2*10.f + 2.f * (1.f + 1.5f*1.f);
    vec4 dx = vec4(1,0,0,0) * (point_size / resolution.x);
    vec4 dy = vec4(0,1,0,0) * (point_size / resolution.y);

    vec4 vertex_offsets[4] = {(-dx-dy), (+dx-dy), (-dx+dy), (+dx+dy)};
    vec2 texcoords[4] = {{0,0}, {1,0}, {0,1}, {1,1}};

    // Add point coordinates
    for (int i = 0; i < 4; ++i)
    {
        gl_Position = gl_in[0].gl_Position + vertex_offsets[i];
        point_uv = texcoords[i];
        EmitVertex();
    }
    EndPrimitive();
}
