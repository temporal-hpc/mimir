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
    gl_Position = proj * view * vec4(2 * (pos / extent) - 1, 1);
}
