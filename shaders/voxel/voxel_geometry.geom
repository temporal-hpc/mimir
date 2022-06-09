#version 450

layout (points) in;
layout (triangle_strip, max_vertices = 24/*12*/) out;

layout (location = 0) in vec3 in_pos[1];
layout (location = 1) in vec3 in_color[1];
layout (location = 2) in float in_alpha[1];

layout (location = 0) out vec3 out_normal;
layout (location = 1) out vec3 out_color;
layout (location = 2) out float out_alpha;

layout (binding = 4) uniform ModelViewProjection
{
  mat4 model;
  mat4 view;
  mat4 proj;
} mvp;

/*layout (binding = 1) uniform Params
{
  vec3 block_size;
} params;*/

void add_face(vec4 center, vec4 shift, vec4 dy, vec4 dx, vec3 n)
{
  vec4 v1 = (center + shift) + (dx - dy);
  vec4 v2 = (center + shift) + (-dx - dy);
  vec4 v3 = (center + shift) + (dx + dy);
  vec4 v4 = (center + shift) + (-dx + dy);

  // In orthographic projection we have to fix our origin (center),
  // because every ray has the same direction
  if (mvp.proj[3][3] == 1.f)
  {
    center = vec4(0.f, 0.f, -1.f, 1.f);
  }

  // Emit a primitive only if the sign of the dot product is positive
  vec4 normal = (mvp.proj * mvp.view * mvp.model * vec4(n, 0.f));

  //if (dot(-center.xyz, normal.xyz) > 0.f)
  //{
    out_normal = normal.xyz;
    out_color = in_color[0];
    out_alpha = in_alpha[0];

    gl_Position = mvp.proj * v1;
    EmitVertex();

    gl_Position = mvp.proj * v2;
    EmitVertex();

    gl_Position = mvp.proj * v3;
    EmitVertex();

    gl_Position = mvp.proj * v4;
    EmitVertex();

    EndPrimitive();
  //}
}

void main()
{
  mat4 model_view = mvp.view * mvp.model;
  vec4 center = model_view * vec4(in_pos[0], 1.f);
  vec3 half_block = .5f * vec3(.1f);//params.block_size;

  vec4 dx = model_view[0] * half_block.x;
  vec4 dy = model_view[1] * half_block.y;
  vec4 dz = model_view[2] * half_block.z;

  add_face(center, +dx, dy, dz, vec3(1.f, 0.f, 0.f));  // Right
  add_face(center, -dx, dz, dy, vec3(-1.f, 0.f, 0.f)); // Left
  add_face(center, +dy, dz, dx, vec3(0.f, 1.f, 0.f));  // Top
  add_face(center, -dy, dx, dz, vec3(0.f, -1.f, 0.f)); // Bottom
  add_face(center, +dz, dx, dy, vec3(0.f, 0.f, 1.f));  // Front
  add_face(center, -dz, dy, dx, vec3(0.f, 0.f, -1.f)); // Back
}
