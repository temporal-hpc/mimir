#include "pack.cuh"
#include "validation.hpp" // checkCuda

__global__ void packPositionsKernel(const float4 *__restrict__ in, float3 *__restrict__ out,
    unsigned int body_count)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= body_count) { return; }
    float4 p = in[i];
    out[i] = make_float3(p.x, p.y, p.z);
}

void packPositionsVec3(const float4 *d_pos4, float *d_pos3, unsigned int body_count,
    int block_size)
{
    unsigned int num_blocks = (body_count + block_size - 1) / block_size;
    packPositionsKernel<<<num_blocks, block_size>>>(
        d_pos4, reinterpret_cast<float3 *>(d_pos3), body_count
    );
    checkCuda(cudaGetLastError());
}
