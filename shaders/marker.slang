import uniforms;

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> scene;
ConstantBuffer<PrimitiveParams> primitive;

static const float SQRT_2 = 1.4142135623730951;
static const float u_antialias = 1;
static const float v_linewidth = 1;

interface ICoord
{
    float4 getVertex(int3 extent);
};

struct MarkerInput<T, let N : int>
{
    vector<T,N> pos : POSITION;
}

struct Float2 : ICoord
{
    float2 pos : POSITION;
    float4 getVertex(int3 extent)
    {
        return getVertexPos2d(pos, extent);
    }
};

struct Double2 : ICoord
{
    MarkerInput<double,2> input;
    float4 getVertex(int3 extent)
    {
        float2 pos = convertDouble<2>(input.pos);
        return getVertexPos2d(pos, extent);
    }
};

struct Half2 : ICoord
{
    MarkerInput<half,2> input;
    float4 getVertex(int3 extent)
    {
        float2 pos = convertHalf<2>(input.pos);
        return getVertexPos2d(pos, extent);
    }
};

struct Int2 : ICoord
{
    MarkerInput<int,2> input;
    float4 getVertex(int3 extent)
    {
        float2 pos = convertInt<2>(input.pos);
        return getVertexPos2d(pos, extent);
    }
};

struct Float3 : ICoord
{
    MarkerInput<float,3> input;
    float4 getVertex(int3 extent)
    {
        return getVertexPos3d(input.pos, extent);
    }
}

struct Double3 : ICoord
{
    MarkerInput<double,3> input;
    float4 getVertex(int3 extent)
    {
        float3 pos = convertDouble<3>(input.pos);
        return getVertexPos3d(pos, extent);
    }
};

struct Half3 : ICoord
{
    MarkerInput<half,3> input;
    float4 getVertex(int3 extent)
    {
        float3 pos = convertHalf<3>(input.pos);
        return getVertexPos3d(pos, extent);
    }
};

struct Int3 : ICoord
{
    MarkerInput<int,3> input;
    float4 getVertex(int3 extent)
    {
        float3 pos = convertInt<3>(input.pos);
        return getVertexPos3d(pos, extent);
    }
};

float4 getVertexPos2d(float2 pos, int3 extent)
{
    return float4(2 * (pos / extent.xy) - 1, 0, 1);
};

float4 getVertexPos3d(float3 pos, int3 extent)
{
    return float4(2 * (pos / extent) - 1, 1);
};

vector<float,N> convertInt<let N:int>(vector<int,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}

vector<float,N> convertHalf<let N:int>(vector<half,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}

vector<float,N> convertFloat<let N:int>(vector<float,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}

vector<float,N> convertDouble<let N:int>(vector<double,N> vec)
{
    vector<float,N> res;
    [unroll]
    for (int i = 0; i < N; ++i) { res[i] = float(vec[i]); }
    return res;
}

struct VertexData
{
    float4 center : POSITION;
    //float4 pos    : SV_Position;
    //float4 color  : COLOR;
    //float marker_size;
};

[shader("vertex")]
VertexData vertexMain<T:ICoord>(T input)
{
    VertexData output;
    output.center = input.getVertex(scene.extent);
    return output;
}

struct MarkerData
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD;
    float size : PSIZE;
    //float4 color : COLOR;
    //float3 normal : NORMAL;
};

[shader("geometry")]
[maxvertexcount(4)]
void geometryMain(point VertexData input[1], inout TriangleStream<MarkerData> stream)
{
    float4x4 view_proj = mul(mvp.view, mvp.proj);
    float4 center = mul(input[0].center, view_proj);
    //float4 center = input[0].center;

    MarkerData output;
    output.size = SQRT_2*primitive.size  + 2 * (v_linewidth + 1.5*u_antialias);

    float4 dx = float4(1,0,0,0) * (output.size / scene.resolution.x);
    float4 dy = float4(0,1,0,0) * (output.size / scene.resolution.y);

    float4 vertex_offsets[4] = {(-dx-dy), (+dx-dy), (-dx+dy), (+dx+dy)};
    float2 texcoords[4] = {{0,0}, {1,0}, {0,1}, {1,1}};

    //output.color = input[0].color;
    //output.normal = normal.xyz;

    // Add point coordinates
    for (int i = 0; i < 4; ++i)
    {
        output.pos = center + vertex_offsets[i];
        output.uv = texcoords[i];
        stream.Append(output);
    }
}

interface IAntialiased
{
    float4 apply(
        float distance,  // Signed distance to line
        float linewidth, // Stroke line width
        float antialias, // Stroke antialiased area
        float4 stroke,   // Stroke color
        float4 fill      // Fill color
    );
}

struct Stroke : IAntialiased
{
    // Fill is unused
    float4 apply(float distance, float linewidth, float antialias, float4 stroke, float4 fill)
    {
        float t = linewidth / 2.0 - antialias;
        float signed_distance = distance;
        float border_distance = abs(signed_distance) - t;
        float alpha = border_distance / antialias;
        alpha = exp(-alpha * alpha);
        if (border_distance < 0.0) return stroke;
        else                       return float4(stroke.rgb, stroke.a * alpha);
    }
}

struct Filled : IAntialiased
{
    // Stroke is unused
    float4 apply(float distance, float linewidth, float antialias, float4 fill, float4 stroke)
    {
        float t = linewidth / 2.0 - antialias;
        float signed_distance = distance;
        float border_distance = abs(signed_distance) - t;
        float alpha = border_distance / antialias;
        alpha = exp(-alpha * alpha);
        if (border_distance < 0.0)      return fill;
        else if (signed_distance < 0.0) return fill;
        else                            return float4(fill.rgb, alpha * fill.a);
    }
}

struct Outline : IAntialiased
{
    float4 apply(float distance, float linewidth, float antialias, float4 stroke, float4 fill)
    {
        float t = linewidth / 2.0 - antialias;
        float signed_distance = distance;
        float border_distance = abs(signed_distance) - t;
        float alpha = border_distance / antialias;
        alpha = exp(-alpha * alpha);
        if (border_distance < 0.0)     return stroke;
        else if(signed_distance < 0.0) return lerp(fill, stroke, sqrt(alpha));
        else                           return float4(stroke.rgb, stroke.a * alpha);
    }
}


interface IShape2D
{
    float apply(float2 p, float size);
}

struct DiscShape : IShape2D
{
    float apply(float2 p, float size)
    {
        return length(p) - size/2;
    }
}

struct SquareShape : IShape2D
{
    float apply(float2 p, float size)
    {
        return max(abs(p.x), abs(p.y)) - size / (2 * SQRT_2);
    }
}

struct TriangleShape : IShape2D
{
    float apply(float2 p, float size)
    {
        float x = SQRT_2/2 * (p.x - p.y);
        float y = SQRT_2/2 * (p.x + p.y);
        float r1 = max(abs(x), abs(y)) - size / (2 * SQRT_2);
        float r2 = p.y;
        return max(r1, r2);
    }
}

struct DiamondShape : IShape2D
{
    float apply(float2 p, float size)
    {
        float x = SQRT_2/2 * (p.x - p.y);
        float y = SQRT_2/2 * (p.x + p.y);
        return max(abs(x), abs(y)) - size / (2 * SQRT_2);
    }
}

interface IShape3D
{
    float2 apply(
        float3 ro, // Ray origin
        float3 rd, // Ray direction
        float3 ce, // Center
        float sz // size
    );
}

struct SphereShape : IShape3D
{
    float2 apply(float3 ro, float3 rd, float3 center, float sz)
    {
        float3 oc = ro - center;
        float b = dot(oc, rd);
        float c = dot(oc, oc) - sz * sz;
        float h = b*b - c;
        return (h<0)? -1 : -b - sqrt(h);
    }
}

struct FragmentOutput
{
    float4 color : SV_Target;
    float depth  : SV_Depth;
};

[shader("fragment")]
FragmentOutput fragmentMain(MarkerData input)
{
    FragmentOutput out;
    
    float2 p = 2 * input.uv - 1;
    // if (dot(p, p) > 1) discard;
    // return primitive.color;
    //float2 p = input.uv - float2(0.5);
    //p = float2(v_rot.x*p.x - v_rot.y*p.y, v_rot.y*p.x + v_rot.x*p.y);
    
    DiscShape shape;
    float distance = shape.apply(p * input.size, primitive.size);
    
    Filled filled;
    out.color = filled.apply(distance, v_linewidth, u_antialias, primitive.color, scene.bg_color);
    //float4x4 view_proj = mul(mvp.view, mvp.proj);
    //float4 clip_pos = mul(view_proj, float4(1, 1, 1, 1));
    //out.depth = clip_pos.z / clip_pos.w;
    out.depth = 1 - out.color.w;
    return out; 
}
