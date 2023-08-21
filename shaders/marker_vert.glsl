#version 460 core

layout (location = 0) in vec3 pos;

layout (std140) uniform Matrices
{
    mat4 proj;
    mat4 view;
};

void main()
{
    vec3 extent = {200, 200, 200};
    vec3 pos2 = (pos / extent) - 1;
    pos2.xy += vec2(.5f, .5f);
    pos2.z -= 1.4;
    gl_Position = proj * view * vec4(pos2, 1);
}
