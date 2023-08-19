#version 460 core

out vec4 FragColor;

in vec2 point_uv;
in float point_size;

vec4 filled(float dist, float linewidth, float antialias, vec4 fill, vec4 stroke)
{
    float t = linewidth / 2.0 - antialias;
    float signed_distance = dist;
    float border_distance = abs(signed_distance) - t;
    float alpha = border_distance / antialias;
    alpha = exp(-alpha * alpha);
    if (border_distance < 0.0)      return fill;
    else if (signed_distance < 0.0) return fill;
    else                            return vec4(fill.rgb, alpha * fill.a);
}

float disc(vec2 p, float size)
{
    return length(p) - size/2;
}

void main()
{
    float primitive_size = 10.f;
    vec4 primitive_color = vec4(0,0,0,1);
    vec4 bg_color = vec4(.5,.5,.5,1);
    
    vec2 p = 2 * point_uv - 1;
    float dist = disc(p * point_size, 10.f);
    
    FragColor = filled(dist, 1.f, 1.f, primitive_color, bg_color);
    gl_FragDepth = 1 - FragColor.w;
}
