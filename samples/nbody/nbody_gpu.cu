#include "nbody_gpu.cuh"

#include <cooperative_groups.h> // cooperative_groups::{sync, this_thread_block}
namespace cg = cooperative_groups;

#include <random>

__constant__ float softening_squared;

cudaError_t setSofteningSquared(float value)
{
    return cudaMemcpyToSymbol(softening_squared, &value, sizeof(float), 0, cudaMemcpyHostToDevice);
}

struct SharedMemory {
    __device__ inline operator float4 *()
    {
        extern __shared__ int __smem[];
        return (float4 *)__smem;
    }

    __device__ inline operator const float4 *() const
    {
        extern __shared__ int __smem[];
        return (float4 *)__smem;
    }
};

__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj)
{
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // dist_sqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float dist_sqr = r.x * r.x + r.y * r.y + r.z * r.z;
    dist_sqr += softening_squared;

    // inv_dist_cube =1/dist_sqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float inv_dist = rsqrtf(dist_sqr);
    float inv_dist_cube = inv_dist * inv_dist * inv_dist;

    // s = m_j * inv_dist_cube [1 FLOP]
    float s = bj.w * inv_dist_cube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

__device__ float3 computeBodyAccel(float4 body_pos, float4 *positions, int num_tiles,
    cg::thread_block cta)
{
    float4 *shared_pos = SharedMemory();
    float3 acc = {0.0f, 0.0f, 0.0f};

    for (int tile = 0; tile < num_tiles; tile++)
    {
        shared_pos[threadIdx.x] = positions[tile * blockDim.x + threadIdx.x];
        cg::sync(cta);

// This is the "tile_calculation" from the GPUG3 article.
#pragma unroll 128
        for (unsigned int counter = 0; counter < blockDim.x; counter++)
        {
            acc = bodyBodyInteraction(acc, body_pos, shared_pos[counter]);
        }
        cg::sync(cta);
    }

    return acc;
}

__global__ void integrateBodies(float4 *__restrict__ new_pos, float4 *__restrict__ old_pos,
    float4 *vel, unsigned int body_count, float delta_time, float damping, int num_tiles)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= body_count) { return; }

    float4 position = old_pos[index];
    float3 accel = computeBodyAccel(position, old_pos, num_tiles, cta);

    // acceleration = force / mass;
    // new velocity = old velocity + acceleration * delta_time
    // note we factor out the body's mass from the equation, here and in bodyBodyInteraction
    // (because they cancel out). Thus here force == acceleration
    float4 velocity = vel[index];

    velocity.x += accel.x * delta_time;
    velocity.y += accel.y * delta_time;
    velocity.z += accel.z * delta_time;

    velocity.x *= damping;
    velocity.y *= damping;
    velocity.z *= damping;

    // new position = old position + velocity * delta_time
    position.x += velocity.x * delta_time;
    position.y += velocity.y * delta_time;
    position.z += velocity.z * delta_time;

    // store new position and velocity
    new_pos[index] = position;
    vel[index] = velocity;
}

void integrateNbodySystem(DeviceData data, unsigned int current_read, float delta_time,
    float damping, unsigned int body_count, int block_size)
{
    int num_blocks = (body_count + block_size - 1) / block_size;
    int num_tiles = (body_count + block_size - 1) / block_size;
    int shmem_size = block_size * 4 * sizeof(float);  // 4 floats for pos

    integrateBodies<<<num_blocks, block_size, shmem_size>>>(
        data.dPos[1 - current_read],
        data.dPos[current_read],
        data.dVel,
        body_count, delta_time, damping, num_tiles
    );

    // check if kernel invocation generated an error
    checkCuda(cudaGetLastError());
}

inline float normalize(float3 &vector)
{
    float dist = sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
    if (dist > 1e-6)
    {
        vector.x /= dist;
        vector.y /= dist;
        vector.z /= dist;
    }

    return dist;
}

inline float dot(float3 v0, float3 v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

inline float3 cross(float3 v0, float3 v1)
{
    float3 rt;
    rt.x = v0.y * v1.z - v0.z * v1.y;
    rt.y = v0.z * v1.x - v0.x * v1.z;
    rt.z = v0.x * v1.y - v0.y * v1.x;
    return rt;
}

void randomizeBodies(NBodyConfig config, float *pos, float *vel, float *color,
    float cluster_scale, float velocity_scale, int body_count, bool vec4vel)
{
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> rand_pos(-1.f, 1.f);
    switch (config)
    {
        default:
        case NBodyConfig::Random:
        {
            float scale = cluster_scale * std::max<float>(1.0f, body_count / (1024.0f));
            float vscale = velocity_scale * scale;

            int p = 0, v = 0;
            int i = 0;

            while (i < body_count)
            {
                float3 point;
                // const int scale = 16;
                point.x = rand_pos(rng);
                point.y = rand_pos(rng);
                point.z = rand_pos(rng);
                float len_sqr = dot(point, point);

                if (len_sqr > 1) continue;

                float3 velocity;
                velocity.x = rand_pos(rng);
                velocity.y = rand_pos(rng);
                velocity.z = rand_pos(rng);
                len_sqr = dot(velocity, velocity);

                if (len_sqr > 1) continue;

                pos[p++] = point.x * scale;  // pos.x
                pos[p++] = point.y * scale;  // pos.y
                pos[p++] = point.z * scale;  // pos.z
                pos[p++] = 1.0f;             // mass

                vel[v++] = velocity.x * vscale;  // pos.x
                vel[v++] = velocity.y * vscale;  // pos.x
                vel[v++] = velocity.z * vscale;  // pos.x

                if (vec4vel) vel[v++] = 1.0f;  // inverse mass

                i++;
            }
        } break;

        case NBodyConfig::Shell:
        {
            float scale = cluster_scale;
            float vscale = scale * velocity_scale;
            float inner = 2.5f * scale;
            float outer = 4.0f * scale;

            int p = 0, v = 0;
            int i = 0;

            while (i < body_count)
            {
                float x, y, z;
                x = rand_pos(rng);
                y = rand_pos(rng);
                z = rand_pos(rng);

                float3 point = {x, y, z};
                float len = normalize(point);
                if (len > 1) { continue; }

                pos[p++] = point.x * (inner + (outer - inner) * rand() / (float)RAND_MAX);
                pos[p++] = point.y * (inner + (outer - inner) * rand() / (float)RAND_MAX);
                pos[p++] = point.z * (inner + (outer - inner) * rand() / (float)RAND_MAX);
                pos[p++] = 1.0f;

                x = 0.0f;  // * (rand() / (float) RAND_MAX * 2 - 1);
                y = 0.0f;  // * (rand() / (float) RAND_MAX * 2 - 1);
                z = 1.0f;  // * (rand() / (float) RAND_MAX * 2 - 1);
                float3 axis = {x, y, z};
                normalize(axis);

                if (1 - dot(point, axis) < 1e-6)
                {
                    axis.x = point.y;
                    axis.y = point.x;
                    normalize(axis);
                }

                // if (point.y < 0) axis = scalevec(axis, -1);
                float3 vv = {pos[4 * i], pos[4 * i + 1], pos[4 * i + 2]};
                vv = cross(vv, axis);
                vel[v++] = vv.x * vscale;
                vel[v++] = vv.y * vscale;
                vel[v++] = vv.z * vscale;

                if (vec4vel) { vel[v++] = 1.0f; }

                i++;
            }
        } break;

        case NBodyConfig::Expand:
        {
            float scale = cluster_scale * body_count / (1024.f);

            if (scale < 1.0f) { scale = cluster_scale; }
            float vscale = scale * velocity_scale;
            int p = 0, v = 0;

            for (int i = 0; i < body_count;)
            {
                float3 point;
                point.x = rand_pos(rng);
                point.y = rand_pos(rng);
                point.z = rand_pos(rng);

                float len_sqr = dot(point, point);
                if (len_sqr > 1) { continue; }

                pos[p++] = point.x * scale;   // pos.x
                pos[p++] = point.y * scale;   // pos.y
                pos[p++] = point.z * scale;   // pos.z
                pos[p++] = 1.0f;              // mass
                vel[v++] = point.x * vscale;  // pos.x
                vel[v++] = point.y * vscale;  // pos.x
                vel[v++] = point.z * vscale;  // pos.x

                if (vec4vel) vel[v++] = 1.0f;  // inverse mass

                i++;
            }
        } break;
    }

    if (color != nullptr)
    {
        std::uniform_real_distribution<float> rand_color(0, 1);
        int v = 0;
        for (int i = 0; i < body_count; i++) {
            color[v++] = rand_color(rng);
            color[v++] = rand_color(rng);
            color[v++] = rand_color(rng);
            color[v++] = 1.0f;
        }
    }
}