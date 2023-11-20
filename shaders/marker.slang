import uniforms;

ConstantBuffer<ModelViewProjection> mvp;
ConstantBuffer<SceneParams> scene;
ConstantBuffer<ViewParams> view;

static const float SQRT_2 = 1.4142135623730951;
static const float u_antialias = 1;
static const float v_linewidth = 1;

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

interface IPosition
{
    float4 getPosition(int3 extent);
};

struct VectorInput<T, let N : int>
{
    vector<T,N> data;
};

struct PositionFloat2 : IPosition
{
    VectorInput<float,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertFloat<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionDouble2 : IPosition
{
    VectorInput<double,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertDouble<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionHalf2 : IPosition
{
    VectorInput<half,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertHalf<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionInt2 : IPosition
{
    VectorInput<int,2> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float2 data = convertInt<2>(input.data);
        return getPosition2d(data, extent);
    }
};

struct PositionFloat3 : IPosition
{
    VectorInput<float,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        return getPosition3d(input.data, extent);
    }
};

struct PositionDouble3 : IPosition
{
    VectorInput<double,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float3 data = convertDouble<3>(input.data);
        return getPosition3d(data, extent);
    }
};

struct PositionHalf3 : IPosition
{
    VectorInput<half,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float3 data = convertHalf<3>(input.data);
        return getPosition3d(data, extent);
    }
};

struct PositionInt3 : IPosition
{
    VectorInput<int,3> input : POSITION;
    float4 getPosition(int3 extent)
    {
        float3 data = convertInt<3>(input.data);
        return getPosition3d(data, extent);
    }
};

float4 getPosition2d(float2 data, int3 extent)
{
    return float4(2 * (data / extent.xy) - 1, 0, 1);
}

float4 getPosition3d(float3 data, int3 extent)
{
    return float4(2 * (data / extent) - 1, 1);
}

interface IColor
{
    float4 getColor();
};

struct ColorDefault : IColor
{
    float4 getColor() { return view.default_color; }
};


struct ColorFloat2 : IColor
{
    VectorInput<float,2> input : POSITION;
    float4 getColor()
    {
        float2 data = convertFloat<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorDouble2 : IColor
{
    VectorInput<double,2> input : POSITION;
    float4 getColor()
    {
        float2 data = convertDouble<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorHalf2 : IColor
{
    VectorInput<half,2> input : POSITION;
    float4 getColor()
    {
        float2 data = convertHalf<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorInt2 : IColor
{
    VectorInput<int,2> input : POSITION;
    float4 getColor()
    {
        float2 data = convertInt<2>(input.data);
        return getColorVec2(data);
    }
};

struct ColorFloat3 : IColor
{
    VectorInput<float,3> input : POSITION;
    float4 getColor()
    {
        return getColorVec3(input.data);
    }
};

struct ColorDouble3 : IColor
{
    VectorInput<double,3> input : POSITION;
    float4 getColor()
    {
        float3 data = convertDouble<3>(input.data);
        return getColorVec3(data);
    }
};

struct ColorHalf3 : IColor
{
    VectorInput<half,3> input : POSITION;
    float4 getColor()
    {
        float3 data = convertHalf<3>(input.data);
        return getColorVec3(data);
    }
};

struct ColorInt3 : IColor
{
    VectorInput<int,3> input : POSITION;
    float4 getColor()
    {
        float3 data = convertInt<3>(input.data);
        return getColorVec3(data);
    }
};

float4 getColorVec2(float2 data) { return float4(data, 0, 1); }
float4 getColorVec3(float3 data) { return float4(data, 1); }

interface ISize
{
    float getSize();
};

struct SizeInt : ISize
{
    VectorInput<int,1> input : PSIZE;
    float getSize() { return convertInt<1>(input.data).x; }
};

struct SizeFloat : ISize
{
    VectorInput<float,1> input : PSIZE;
    float getSize() { return convertFloat<1>(input.data).x; }
};

struct SizeDouble : ISize
{
    VectorInput<double,1> input : PSIZE;
    float getSize() { return convertDouble<1>(input.data).x; }
};

struct SizeHalf : ISize
{
    VectorInput<half,1> input : PSIZE;
    float getSize() { return convertHalf<1>(input.data).x; }
};

struct SizeDefault : ISize
{
    float getSize() { return view.default_size; }
};

struct VertexData
{
    float4 center : POSITION;
    float4 color  : COLOR;
    float size    : PSIZE;
};

[shader("vertex")]
VertexData vertexMain<P:IPosition, C:IColor, S:ISize>(P in_pos, C in_color, S in_size)
{
    VertexData output;
    output.center = in_pos.getPosition(scene.extent);
    output.color  = in_color.getColor();
    output.size   = in_size.getSize();
    return output;
}

struct MarkerData
{
    float4 pos   : SV_Position;
    float2 uv    : TEXCOORD;
    float4 color : COLOR;
    float size   : PSIZE;
    float ele_size;
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
    output.size = SQRT_2*input[0].size  + 2 * (v_linewidth + 1.5*u_antialias);

    float4 dx = float4(1,0,0,0) * (output.size / scene.resolution.x);
    float4 dy = float4(0,1,0,0) * (output.size / scene.resolution.y);

    float4 vertex_offsets[4] = {(-dx-dy), (+dx-dy), (-dx+dy), (+dx+dy)};
    float2 texcoords[4] = {{0,0}, {1,0}, {0,1}, {1,1}};

    output.color = input[0].color;
    output.ele_size = input[0].size;
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
    // return element.color;
    //float2 p = input.uv - float2(0.5);
    //p = float2(v_rot.x*p.x - v_rot.y*p.y, v_rot.y*p.x + v_rot.x*p.y);
    
    DiscShape shape;
    float distance = shape.apply(p * input.size, input.ele_size);
    
    Filled filled;
    out.color = filled.apply(distance, v_linewidth, u_antialias, input.color, scene.bg_color);
    //float4x4 view_proj = mul(mvp.view, mvp.proj);
    //float4 clip_pos = mul(view_proj, float4(1, 1, 1, 1));
    //out.depth = clip_pos.z / clip_pos.w;
    out.depth = 1 - out.color.w;
    return out; 
}
