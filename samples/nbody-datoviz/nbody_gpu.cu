#include "nbody_gpu.cuh"

#include <cooperative_groups.h> // cooperative_groups::{sync, this_thread_block}
namespace cg = cooperative_groups;

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
