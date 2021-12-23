#version 450

layout(binding = 0) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;

layout(location = 0) out vec2 out_uv;

// Render a fullscren texture without buffers
// Uses a single triangle instead of a 2-triangle quad
void main()
{
  out_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = mvp.proj * mvp.view * mvp.model * vec4(out_uv * 2.f - 1.f, 1.f, 1.f);
}
