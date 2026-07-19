#include "mimir/voxel_lod.hpp"
namespace mimir {
namespace { constexpr int kThreads = 256;
__global__ void poolMaxKernel(const int* fine, uint32_t N, int* coarse, uint32_t M){
    uint64_t mCells=(uint64_t)M*M*M;
    for(uint64_t c=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;c<mCells;c+=(uint64_t)blockDim.x*gridDim.x){
        uint32_t cx=(uint32_t)(c%M), cy=(uint32_t)((c/M)%M), cz=(uint32_t)(c/((uint64_t)M*M));
        uint32_t x0=(uint32_t)((uint64_t)cx*N/M),x1=(uint32_t)(((uint64_t)cx+1)*N/M);
        uint32_t y0=(uint32_t)((uint64_t)cy*N/M),y1=(uint32_t)(((uint64_t)cy+1)*N/M);
        uint32_t z0=(uint32_t)((uint64_t)cz*N/M),z1=(uint32_t)(((uint64_t)cz+1)*N/M);
        int m=0; for(uint32_t z=z0;z<z1;++z)for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x)
            m=max(m,fine[(uint64_t)x+N*((uint64_t)y+(uint64_t)N*z)]);
        coarse[c]=m;
    }
}}
void voxelPoolMax(const int* fine,uint32_t N,int* coarse,uint32_t M,cudaStream_t s){
    uint64_t b=((uint64_t)M*M*M+kThreads-1)/kThreads; if(b>2147483647ull)b=2147483647ull; if(b<1)b=1;
    poolMaxKernel<<<(uint32_t)b,kThreads,0,s>>>(fine,N,coarse,M);
}

namespace {
__global__ void compactLivingKernel(const int* state, uint32_t N, float3 origin, float3 spacing,
    float* out_positions, uint32_t capacity, uint32_t* counter)
{
    uint64_t cells = (uint64_t)N * N * N;
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < cells;
         i += (uint64_t)blockDim.x * gridDim.x)
    {
        if (state[i] == 0) { continue; }
        uint32_t idx = atomicAdd(counter, 1u);
        if (idx >= capacity) { continue; } // overflow guard (host sizes capacity >= living count)
        uint32_t x = (uint32_t)(i % N);
        uint32_t y = (uint32_t)((i / N) % N);
        uint32_t z = (uint32_t)(i / ((uint64_t)N * N));
        out_positions[3 * idx + 0] = origin.x + (float)x * spacing.x;
        out_positions[3 * idx + 1] = origin.y + (float)y * spacing.y;
        out_positions[3 * idx + 2] = origin.z + (float)z * spacing.z;
    }
}
}
void voxelCompactLiving(const int* state, uint32_t N, float3 origin, float3 spacing,
    float* out_positions, uint32_t capacity, uint32_t* d_count, cudaStream_t s)
{
    cudaMemsetAsync(d_count, 0, sizeof(uint32_t), s);
    uint64_t cells = (uint64_t)N * N * N;
    uint64_t b = (cells + kThreads - 1) / kThreads;
    if (b > 65535ull) b = 65535ull; if (b < 1) b = 1; // grid-stride loop covers the rest
    compactLivingKernel<<<(uint32_t)b, kThreads, 0, s>>>(
        state, N, origin, spacing, out_positions, capacity, d_count);
}}
